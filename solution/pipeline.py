"""
More Than A Hack 2027 — Improved Possession Pipeline
======================================================

Improvements over the reference baseline (spellF1 floor: 0.2433):

  1. Smart dead-ball detection
       - Collective player speed drop  → real referee whistles
       - Smart Tagging set pieces      → dead-ball windows around restarts
       - Ball out-of-pitch boundaries  → ball has left the field
  2. Short-occlusion bridging
       - If the ball disappears for <1.5 s during active play, we keep
         the previous live-play label instead of declaring D.
  3. Possession persistence state machine
       - Possession is sticky: it stays with the last controlling team
         until a controlled take-over or a dead ball occurs.
       - A "controlled take-over" needs N_CTRL_FRAMES consecutive frames
         where the nearest player belongs to the other team AND the ball
         is visible (not just a gap).
  4. Minimum spell length
       - Spells shorter than MIN_SPELL_FRAMES are absorbed into their
         neighbours to prevent microscopic spurious transitions.

Usage (from repo root):
    python solution/pipeline.py
    python evaluation/run_evaluation.py
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]   # metres; 105.3 × 68.0
FPS = GAMES[GAME]["fps"]                            # 25

# Team codes (internal)
HOME, AWAY, NOBODY = 0, 1, -1
LABEL_STR = {HOME: "H", AWAY: "A", NOBODY: "D"}

# Dead-ball detection
SPEED_DEAD_THRESHOLD  = 1.1   # m/s — mean team speed below this → dead ball
SPEED_ALIVE_THRESHOLD = 1.6   # m/s — above this confirms active play
SPEED_WINDOW          = 20    # frames to average speed over (0.8 s)

# Possession persistence
OCCLUSION_BRIDGE_FRAMES = 37  # bridge ball gaps up to 1.5 s (37 frames @ 25 fps)
N_CTRL_FRAMES           = 8   # consecutive frames to flip possession

# Minimum spell length (frames) — spells shorter than this are merged away
MIN_SPELL_FRAMES = 50         # 2 seconds @ 25 fps

# Output
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
SUBMISSION_NAME = "solution_v2"

# Graded half boundaries (video frames) — from DATA.md
HALF_BLOCKS = [(11185, 90651), (110808, 190197)]

# ---------------------------------------------------------------------------
# Step 1 — Read ATD raw
# ---------------------------------------------------------------------------

FIELD_SEP  = ":"
GROUP_SEP  = ";"
MISSING    = "N"


def team_of_each_track(meta_path):
    """Array[n_slots] → HOME/AWAY for each track slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id  = team_ids[0]
    slots    = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def read_atd(raw_path, team_of_track, n_frames):
    """
    Stream the raw ATD once, returning:
        ball_x     (n_frames,)  float32  NaN where invisible
        ball_y     (n_frames,)  float32  NaN where invisible
        nearest    (n_frames,)  int8     HOME/AWAY/NOBODY (nearest player to ball)
        mean_speed (n_frames,)  float32  mean speed of all visible players (m/s)
    """
    ball_x    = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y    = np.full(n_frames, np.nan, dtype=np.float32)
    nearest   = np.full(n_frames, NOBODY, dtype=np.int8)
    tot_speed = np.zeros(n_frames, dtype=np.float32)
    n_players = np.zeros(n_frames, dtype=np.int16)

    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            idx      = frame_id - 1          # ATD counts from 1, submission from 0
            if idx < 0 or idx >= n_frames:
                continue

            # ---- ball ----
            ball_str = parts[2]
            has_ball = ball_str[0] != MISSING
            if has_ball:
                bcomma1 = ball_str.index(",")
                bcomma2 = ball_str.index(",", bcomma1 + 1)
                bx = float(ball_str[:bcomma1])  * PITCH_LENGTH
                by = float(ball_str[bcomma1+1:bcomma2]) * PITCH_WIDTH
                ball_x[idx] = bx
                ball_y[idx] = by

            # ---- players ----
            best_dist = None
            best_team = NOBODY
            for slot, grp in enumerate(parts[1].split(GROUP_SEP)):
                if grp[0] == MISSING:
                    continue
                c1 = grp.index(",")
                c2 = grp.index(",", c1 + 1)
                px = float(grp[:c1])      * PITCH_LENGTH
                py = float(grp[c1+1:c2]) * PITCH_WIDTH
                spd = float(grp[c2+1:])  # already in m/s

                n_players[idx] += 1
                tot_speed[idx] += spd

                if has_ball and slot < len(team_of_track):
                    dx = px - bx;  dy = py - by
                    dist = dx*dx + dy*dy
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_team = team_of_track[slot]

            nearest[idx] = best_team

    # mean speed per frame (m/s)
    with np.errstate(invalid="ignore"):
        mean_speed = np.where(n_players > 0, tot_speed / n_players, 0.0).astype(np.float32)

    return ball_x, ball_y, nearest, mean_speed


# ---------------------------------------------------------------------------
# Step 2 — Smart Tagging dead-ball windows
# ---------------------------------------------------------------------------

SET_PIECE_CODES = {
    "set pieces", "dead ball"
}


def load_dead_ball_intervals(xml_path, n_frames, margin_before=50, margin_after=0):
    """
    Returns a boolean array dead_from_tags[n_frames]:
    True wherever a Smart Tagging set-piece is APPROACHING (before the restart).
    
    Key insight: a set-piece ENDS when play restarts (ball is struck). We mark
    the period BEFORE the event as dead-ball, not after it, because after the
    kick the ball is live and possession belongs to the restarting team.
    
    margin_before: frames to mark before the set-piece start (default = 2.0 s)
    margin_after:  frames into the set-piece to still mark as dead (default = 0)
                   Set to 0 so we don't consume live play after the restart.
    """
    dead = np.zeros(n_frames, dtype=bool)

    root = ET.parse(xml_path).getroot()
    fps  = GAMES[GAME]["fps"]

    for inst in root.iter("instance"):
        code  = (inst.findtext("code") or "").strip().lower()
        start = inst.findtext("start")
        end   = inst.findtext("end")
        if code not in SET_PIECE_CODES or start is None or end is None:
            continue

        f_start = max(0, int(round(float(start) * fps)) - margin_before)
        # Only mark up to the START of the set piece (the kick restarts play)
        f_end   = min(n_frames, int(round(float(start) * fps)) + margin_after)
        dead[f_start:f_end] = True

    return dead


# ---------------------------------------------------------------------------
# Step 3 — Collective-speed dead-ball detection
# ---------------------------------------------------------------------------

def speed_dead_mask(mean_speed, n_frames):
    """
    Smooth mean speed over SPEED_WINDOW frames, then threshold.
    Returns bool array: True where collective speed is below SPEED_DEAD_THRESHOLD.
    """
    # Cumulative-sum smoothing (same trick as baseline.smooth)
    cum = np.concatenate(([0.0], np.cumsum(mean_speed)))
    idx = np.arange(n_frames)
    low = np.maximum(0,        idx - SPEED_WINDOW)
    hi  = np.minimum(n_frames, idx + SPEED_WINDOW + 1)
    win = (hi - low).astype(np.float32)
    win = np.where(win == 0, 1, win)
    smoothed = (cum[hi] - cum[low]) / win

    return smoothed < SPEED_DEAD_THRESHOLD


# ---------------------------------------------------------------------------
# Step 4 — Pitch-boundary dead ball
# ---------------------------------------------------------------------------

def boundary_dead_mask(ball_x, ball_y, margin_m=1.0):
    """True where the ball is tracked but outside the pitch boundaries."""
    has_ball = ~np.isnan(ball_x)
    out_x    = (ball_x < -margin_m) | (ball_x > PITCH_LENGTH + margin_m)
    out_y    = (ball_y < -margin_m) | (ball_y > PITCH_WIDTH  + margin_m)
    return has_ball & (out_x | out_y)


# ---------------------------------------------------------------------------
# Step 5 — Combine dead-ball signals
# ---------------------------------------------------------------------------

def combined_dead_mask(ball_x, ball_y, mean_speed, tag_dead, n_frames):
    spd_dead  = speed_dead_mask(mean_speed, n_frames)
    bnd_dead  = boundary_dead_mask(ball_x, ball_y)

    # Combine: smart-tagging is authoritative; speed drop adds more
    dead = tag_dead | bnd_dead | spd_dead
    return dead


# ---------------------------------------------------------------------------
# Step 6 — Possession state machine with occlusion bridging
# ---------------------------------------------------------------------------

def possession_state_machine(nearest, dead_mask, n_frames):
    """
    State machine that produces per-frame labels:
        H  home has the ball (alive)
        A  away has the ball (alive)
        D  dead ball

    Rules:
      - When dead_mask is True, output D.
      - When ball is visible (nearest != NOBODY) and play is alive:
          count how many consecutive frames the other team is nearest.
          After N_CTRL_FRAMES consecutive frames, flip possession.
      - When ball is invisible (nearest == NOBODY) and play is alive:
          if the gap is short (<= OCCLUSION_BRIDGE_FRAMES), hold the
          current possession (passes, headers, occlusions in the air).
          if the gap is long, hold possession still (no info = no flip).
    """
    labels = np.empty(n_frames, dtype="<U1")
    labels[:] = "X"

    # Determine graded frames (inside the two halves)
    in_half = np.zeros(n_frames, dtype=bool)
    for start, stop in HALF_BLOCKS:
        in_half[start:stop+1] = True

    state   = NOBODY   # current possessing team
    streak  = 0        # consecutive frames with the other team nearest
    occlusion_len = 0  # consecutive invisible frames in live play

    for f in range(n_frames):
        # Frames outside the halves get X (never scored)
        if not in_half[f]:
            labels[f] = "X"
            streak     = 0
            occlusion_len = 0
            continue

        if dead_mask[f]:
            labels[f]     = "D"
            state         = NOBODY
            streak        = 0
            occlusion_len = 0
            continue

        # Active play frame
        near = int(nearest[f])

        if near == NOBODY:
            # Ball not visible
            occlusion_len += 1
            streak = 0
            # Hold the current state; don't declare D just because we can't see the ball
            labels[f] = LABEL_STR.get(state, "D")
        else:
            # Ball visible
            occlusion_len = 0

            if state == NOBODY:
                # First visible frame after dead ball — immediately take the nearest player
                state  = near
                streak = 0
            elif near == state:
                # Same team → reset opposing streak
                streak = 0
            else:
                # Other team is nearest
                streak += 1
                if streak >= N_CTRL_FRAMES:
                    # Controlled take-over
                    state  = near
                    streak = 0

            labels[f] = LABEL_STR.get(state, "D")

    return labels


# ---------------------------------------------------------------------------
# Step 7 — Minimum spell-length enforcement
# ---------------------------------------------------------------------------

def enforce_min_spell(labels, n_frames, min_frames=MIN_SPELL_FRAMES):
    """
    Remove micro-spells shorter than min_frames by absorbing them into
    the adjacent (preceding) spell.  Works within each half separately
    so halftime gaps are never bridged.
    """
    out = labels.copy()

    for start_h, stop_h in HALF_BLOCKS:
        window = out[start_h:stop_h+1]
        L = len(window)
        if L == 0:
            continue

        # Find spell boundaries
        changed = True
        while changed:
            changed = False
            i = 0
            while i < L:
                spell_lbl = window[i]
                if spell_lbl == "X":
                    i += 1
                    continue
                j = i
                while j < L and window[j] == spell_lbl:
                    j += 1
                spell_len = j - i
                if spell_len < min_frames:
                    # Absorb into the preceding label
                    if i > 0:
                        prev_lbl = window[i - 1]
                    else:
                        prev_lbl = window[j] if j < L else "D"
                    window[i:j] = prev_lbl
                    changed = True
                i = j

        out[start_h:stop_h+1] = window

    return out


# ---------------------------------------------------------------------------
# Step 8 — Write submission CSV
# ---------------------------------------------------------------------------

def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(f, lbl) for f, lbl in enumerate(labels))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Improved possession pipeline")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for the CSV")
    parser.add_argument("--name", default=SUBMISSION_NAME, help="submission filename stem")
    args = parser.parse_args()

    n_frames = GAMES[GAME]["n_video_frames"]

    print("=== More Than A Hack 2027 — Improved Pipeline ===\n")

    # -- Load data --
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    xml_path            = require(["smart_tagging"])[0]

    print("Loading team-track mapping …")
    team_of_track = team_of_each_track(meta_path)
    print("  {} track slots ({} home, {} away)".format(
        len(team_of_track),
        int((team_of_track == HOME).sum()),
        int((team_of_track == AWAY).sum()),
    ))

    print("Streaming {} ATD frames …".format(n_frames))
    ball_x, ball_y, nearest, mean_speed = read_atd(raw_path, team_of_track, n_frames)
    ball_visible = int(np.sum(~np.isnan(ball_x)))
    print("  Ball visible on {} / {} frames ({:.1f}%)".format(
        ball_visible, n_frames, 100.0 * ball_visible / n_frames))
    print("  Mean player speed overall: {:.2f} m/s".format(float(mean_speed[mean_speed > 0].mean())))

    print("Loading Smart Tagging dead-ball windows …")
    tag_dead = load_dead_ball_intervals(xml_path, n_frames)
    print("  Dead-from-tags on {} frames ({:.1f}%)".format(
        int(tag_dead.sum()), 100.0 * tag_dead.sum() / n_frames))

    print("Building combined dead-ball mask …")
    dead_mask = combined_dead_mask(ball_x, ball_y, mean_speed, tag_dead, n_frames)
    print("  Dead frames total: {} ({:.1f}%)".format(
        int(dead_mask.sum()), 100.0 * dead_mask.sum() / n_frames))

    print("Running possession state machine …")
    labels = possession_state_machine(nearest, dead_mask, n_frames)

    print("Enforcing minimum spell length ({} frames = {:.1f} s) …".format(
        MIN_SPELL_FRAMES, MIN_SPELL_FRAMES / FPS))
    labels = enforce_min_spell(labels, n_frames)

    # Stats
    h_count = int(np.sum(labels == "H"))
    a_count = int(np.sum(labels == "A"))
    d_count = int(np.sum(labels == "D"))
    x_count = int(np.sum(labels == "X"))
    print("\nLabel distribution:")
    print("  H {:>7}  A {:>7}  D {:>7}  X {:>7}".format(h_count, a_count, d_count, x_count))

    out_path = os.path.join(args.out, "{}.csv".format(args.name))
    write_submission(out_path, labels)
    print("\nWritten → {}".format(os.path.relpath(out_path, ROOT)))
    print("\nScore it:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
