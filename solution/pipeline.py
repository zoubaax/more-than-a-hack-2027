"""
More Than A Hack 2027 — Possession Pipeline
============================================

A clean, fully data-driven possession pipeline. No hardcoded values —
all constants are derived from games.py, the ATD metadata XML, or
explicit tuneable parameters at the top of the file.

Approach
--------
1. Read the Automatic Tracking Data (ATD) in one streaming pass.
   Extract per-frame: ball position, nearest-player team and distance,
   mean collective player speed.

2. Dead-ball detection (three signals, combined):
   a. Smart Tagging set pieces  — the period BEFORE each restart is dead.
   b. Collective player speed   — when everyone slows to a walk (< threshold),
      the referee has whistled.
   c. Ball out-of-pitch boundary — ball tracked but outside the lines.

3. Restart-seeded possession
   Each Smart Tagging set piece carries a Team tag. When play restarts
   (the ball is struck), we immediately seed possession to that team
   instead of waiting N_CTRL_FRAMES proximity frames to discover it.
   Team names are read from the ATD metadata, not hardcoded.

4. Possession state machine
   Possession is sticky. It flips only when the opponent's nearest player
   is within CONTROL_RADIUS_M for N_CTRL_FRAMES consecutive frames.
   Short ball occlusions (passes, headers) are bridged without flipping.

5. Minimum spell-length enforcement
   Micro-spells shorter than MIN_SPELL_FRAMES are absorbed into their
   predecessor to prevent spurious transitions.

Usage (from repo root):
    python solution/pipeline.py
    python evaluation/validate_submission.py evaluation/results/<name>.csv
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

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# ---------------------------------------------------------------------------
# Tuneable parameters — change here, nowhere else
# ---------------------------------------------------------------------------

# Dead-ball detection
SPEED_DEAD_THRESHOLD = 1.05   # m/s  — smoothed collective player speed below this → dead ball
SPEED_WINDOW         = 25     # frames — smoothing half-width (1.0 s at 25 fps)
BOUNDARY_MARGIN_M    = 1.0    # m  — ball must be this far outside the pitch to flag as out

# Restart dead-ball window (Smart Tagging set pieces)
TAG_MARGIN_BEFORE    = 50     # frames — mark dead this many frames BEFORE the restart kick
TAG_SEED_AFTER       = 0      # frames — optional seed window after kick (0 = off)

# Possession state machine
N_CTRL_FRAMES        = 8      # frames — consecutive opponent-nearest frames to flip possession

# Post-processing
MIN_SPELL_FRAMES     = 50     # frames — spells shorter than this are merged (2.0 s at 25 fps)

# Output
RESULTS_DIR     = os.path.join(ROOT, "evaluation", "results")
SUBMISSION_NAME = "solution_v4"

# Internal team codes — ordinal, assigned at load time from the ATD metadata.
# 0 = home (first team in the XML), 1 = away, -1 = unknown / no ball.
HOME, AWAY, NOBODY = 0, 1, -1

# Raw ATD line format: "<frame_id>:<players>:<ball>"
FIELD_SEP = ":"
GROUP_SEP  = ";"
MISSING    = "N"   # first char of "NaN" — faster than float conversion to detect missing values

# ---------------------------------------------------------------------------
# Step 1 — Team metadata from the ATD XML
# ---------------------------------------------------------------------------

def load_team_metadata(meta_path):
    """
    Returns:
        team_of_track (np.int8 array)   — HOME or AWAY for each track slot
        team_names    (dict)             — {HOME: "Barcelona", AWAY: "Atletic Club"}
        tag_name_map  (dict)             — lower-cased Smart Tagging Team values → HOME/AWAY

    Everything is derived from the ATD metadata XML. No club names are
    hardcoded in this module.
    """
    root     = ET.parse(meta_path).getroot()
    teams    = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected exactly 2 teams, found {}".format(
            meta_path, len(teams)))

    # FIFA/EPTS convention: first team = home.
    home_id   = teams[0].get("id")
    home_name = teams[0].findtext("Name") or "Home"
    away_name = teams[1].findtext("Name") or "Away"

    team_names = {HOME: home_name, AWAY: away_name}

    # Build a case-insensitive name map for Smart Tagging Team values.
    # Include every reasonable variant: exact name and each word individually.
    tag_name_map = {}
    for code, name in team_names.items():
        tag_name_map[name.lower()] = code
        for word in name.lower().split():
            if len(word) > 3:   # skip short words like "fc", "cf", "de"
                tag_name_map[word] = code

    # Track-slot → team code
    slots         = [p.get("teamId") for p in root.iter("Player")]
    team_of_track = np.array(
        [HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8
    )

    return team_of_track, team_names, tag_name_map


# ---------------------------------------------------------------------------
# Step 2 — Stream the ATD raw file
# ---------------------------------------------------------------------------

def read_atd(raw_path, team_of_track, n_frames, pitch):
    """
    Single-pass stream of the raw ATD text file.

    Returns (all arrays of length n_frames):
        ball_x     float32  NaN where ball is not tracked
        ball_y     float32  NaN where ball is not tracked
        nearest    int8     HOME / AWAY / NOBODY (nearest tracked player to ball)
        near_dist  float32  metres to nearest player (NaN where no ball)
        mean_speed float32  mean speed of all visible players in m/s
    """
    length, width = pitch
    ball_x    = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y    = np.full(n_frames, np.nan, dtype=np.float32)
    nearest   = np.full(n_frames, NOBODY, dtype=np.int8)
    near_dist = np.full(n_frames, np.nan, dtype=np.float32)
    tot_speed = np.zeros(n_frames, dtype=np.float32)
    n_players = np.zeros(n_frames, dtype=np.int16)

    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            idx      = frame_id - 1          # ATD counts from 1; submission from 0
            if idx < 0 or idx >= n_frames:
                continue

            # ---- ball ----
            ball_str = parts[2]
            has_ball = ball_str[0] != MISSING
            bx = by = None
            if has_ball:
                c1 = ball_str.index(",")
                c2 = ball_str.index(",", c1 + 1)
                bx = float(ball_str[:c1])     * length
                by = float(ball_str[c1+1:c2]) * width
                ball_x[idx] = bx
                ball_y[idx] = by

            # ---- players ----
            best_sq = None
            best_team = NOBODY
            for slot, grp in enumerate(parts[1].split(GROUP_SEP)):
                if grp[0] == MISSING:
                    continue
                c1  = grp.index(",")
                c2  = grp.index(",", c1 + 1)
                px  = float(grp[:c1])      * length
                py  = float(grp[c1+1:c2])  * width
                spd_str = grp[c2+1:]
                if spd_str[0] != MISSING:
                    tot_speed[idx] += float(spd_str)
                    n_players[idx] += 1

                if has_ball and slot < len(team_of_track):
                    sq = (px - bx) ** 2 + (py - by) ** 2
                    if best_sq is None or sq < best_sq:
                        best_sq   = sq
                        best_team = team_of_track[slot]

            nearest[idx] = best_team
            if best_sq is not None:
                near_dist[idx] = best_sq ** 0.5

    with np.errstate(invalid="ignore"):
        mean_speed = np.where(
            n_players > 0, tot_speed / n_players, 0.0
        ).astype(np.float32)

    return ball_x, ball_y, nearest, near_dist, mean_speed


# ---------------------------------------------------------------------------
# Step 3 — Smart Tagging: dead-ball windows + restart possession seeds
# ---------------------------------------------------------------------------

SET_PIECE_CODES = {"set pieces", "dead ball"}


def load_smart_tagging(xml_path, n_frames, tag_name_map, fps=25):
    """
    Parse the Smart Tagging XML and return:
        dead_from_tags (n_frames,) bool   — frames marked dead BEFORE each restart
        restart_team   (n_frames,) int8   — HOME/AWAY seed for frames just AFTER restart

    Key principle:
      A set-piece START marks when the ball is struck and play resumes.
      The dead-ball period is BEFORE that instant, not the whole tagged interval.
      After the kick, possession is seeded to the tagged team for TAG_SEED_AFTER frames.

    Team tags are matched case-insensitively against tag_name_map (built from ATD
    metadata), so no club names are hardcoded here.
    """
    dead         = np.zeros(n_frames, dtype=bool)
    restart_team = np.full(n_frames, NOBODY, dtype=np.int8)

    root = ET.parse(xml_path).getroot()

    for inst in root.iter("instance"):
        code  = (inst.findtext("code") or "").strip().lower()
        start = inst.findtext("start")
        if code not in SET_PIECE_CODES or start is None:
            continue

        f_start = int(round(float(start) * fps))

        # Dead-ball window BEFORE the restart kick
        dead_from = max(0, f_start - TAG_MARGIN_BEFORE)
        dead[dead_from:f_start] = True

        # Identify the restarting team from the Team tag
        tags      = {lbl.findtext("group"): lbl.findtext("text")
                     for lbl in inst.findall("label")}
        team_raw  = (tags.get("Team") or "").strip().lower()
        # Try full name first, then each word
        team_code = tag_name_map.get(team_raw, NOBODY)
        if team_code == NOBODY:
            for word in team_raw.split():
                team_code = tag_name_map.get(word, NOBODY)
                if team_code != NOBODY:
                    break

        # Seed the restarting team for TAG_SEED_AFTER frames after the kick
        if team_code != NOBODY:
            seed_to = min(n_frames, f_start + TAG_SEED_AFTER)
            restart_team[f_start:seed_to] = team_code

    return dead, restart_team


# ---------------------------------------------------------------------------
# Step 4 — Dead-ball masks
# ---------------------------------------------------------------------------

def speed_dead_mask(mean_speed, n_frames):
    """
    Smoothed collective player speed below SPEED_DEAD_THRESHOLD → dead ball.
    Uses a centred cumulative-sum window (O(n), no loop over frames).
    """
    cum = np.concatenate(([0.0], np.cumsum(mean_speed)))
    idx = np.arange(n_frames)
    lo  = np.maximum(0,        idx - SPEED_WINDOW)
    hi  = np.minimum(n_frames, idx + SPEED_WINDOW + 1)
    win = (hi - lo).astype(np.float32)
    win = np.where(win == 0, 1.0, win)
    smoothed = (cum[hi] - cum[lo]) / win
    return smoothed < SPEED_DEAD_THRESHOLD


def boundary_dead_mask(ball_x, ball_y, pitch):
    """Ball tracked but outside the pitch boundary → dead ball."""
    length, width = pitch
    has_ball = ~np.isnan(ball_x)
    out_x    = (ball_x < -BOUNDARY_MARGIN_M) | (ball_x > length + BOUNDARY_MARGIN_M)
    out_y    = (ball_y < -BOUNDARY_MARGIN_M) | (ball_y > width + BOUNDARY_MARGIN_M)
    return has_ball & (out_x | out_y)


def combined_dead_mask(ball_x, ball_y, mean_speed, tag_dead, n_frames, pitch):
    """Union of all three dead-ball signals."""
    spd_dead = speed_dead_mask(mean_speed, n_frames)
    bnd_dead = boundary_dead_mask(ball_x, ball_y, pitch)
    return tag_dead | bnd_dead | spd_dead


# ---------------------------------------------------------------------------
# Step 5 — Possession state machine
# ---------------------------------------------------------------------------

def possession_state_machine(nearest, near_dist, dead_mask, restart_team,
                              n_frames, team_names, half_blocks):
    """
    Per-frame label assignment.

    Label alphabet:
        "H"  home team in possession, ball alive
        "A"  away team in possession, ball alive
        "D"  dead ball (referee has stopped play)
        "X"  outside the graded halves (not scored)

    Decision order per frame:
      1. Outside a graded half     → "X"
      2. dead_mask is True         → "D", reset state
      3. restart_team seed active  → seed team wins, no streak needed
      4. Ball not visible          → hold current state (occlusion / aerial)
      5. Nearest player:
           same team as current state → reset opponent streak
           opponent team              → increment streak; flip after N_CTRL_FRAMES
    """
    label_of = {HOME: "H", AWAY: "A", NOBODY: "D"}

    labels  = np.empty(n_frames, dtype="<U1")
    labels[:] = "X"

    # Build a boolean mask: True where frame is inside a graded half
    in_half = np.zeros(n_frames, dtype=bool)
    for first, last in half_blocks:
        in_half[first:last + 1] = True

    state  = NOBODY
    streak = 0

    for f in range(n_frames):

        # -- outside the match --
        if not in_half[f]:
            labels[f] = "X"
            state     = NOBODY
            streak    = 0
            continue

        # -- dead ball --
        if dead_mask[f]:
            labels[f] = "D"
            state     = NOBODY
            streak    = 0
            continue

        # -- restart seed from Smart Tagging --
        seed = int(restart_team[f])
        if seed != NOBODY:
            state     = seed
            streak    = 0
            labels[f] = label_of[state]
            continue

        # -- active play --
        near = int(nearest[f])

        if near == NOBODY:
            # Ball invisible: hold current state (pass, airborne, occlusion)
            streak    = 0
            labels[f] = label_of.get(state, "D")
        else:
            # Ball visible: nearest player contact candidate
            if state == NOBODY:
                # First possession after dead ball
                state  = near
                streak = 0
            elif near == state:
                streak = 0
            else:
                streak += 1
                if streak >= N_CTRL_FRAMES:
                    state  = near
                    streak = 0

            labels[f] = label_of.get(state, "D")

    return labels


# ---------------------------------------------------------------------------
# Step 6 — Minimum spell-length enforcement
# ---------------------------------------------------------------------------

def enforce_min_spell(labels, n_frames, half_blocks, min_frames=MIN_SPELL_FRAMES):
    """
    Absorb spells shorter than min_frames into their predecessor.
    Operates within each half independently so halftime is never bridged.
    """
    out = labels.copy()

    for first, last in half_blocks:
        window  = out[first:last + 1]
        L       = len(window)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < L:
                lbl = window[i]
                if lbl == "X":
                    i += 1
                    continue
                j = i
                while j < L and window[j] == lbl:
                    j += 1
                if j - i < min_frames:
                    prev        = window[i - 1] if i > 0 else (window[j] if j < L else "D")
                    window[i:j] = prev
                    changed     = True
                i = j

        out[first:last + 1] = window

    return out


# ---------------------------------------------------------------------------
# Step 7 — Write submission CSV
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME,          help="game key (default: %(default)s)")
    parser.add_argument("--out",  default=RESULTS_DIR,   help="output directory for the CSV")
    parser.add_argument("--name", default=SUBMISSION_NAME, help="CSV filename stem")
    args = parser.parse_args()

    # Resolve n_frames, half blocks, pitch, and fps dynamically from games.py
    n_frames   = source_frames("tactical", args.game)
    half_blks  = [(f, l) for _, f, l in half_video_blocks(args.game)]
    pitch      = GAMES[args.game]["pitch"]
    fps        = GAMES[args.game]["fps"]

    print("=== More Than A Hack 2027 — Possession Pipeline ===")
    print("game: {}   frames: {}   halves: {}".format(
        args.game, n_frames,
        " | ".join("{}-{}".format(f, l) for f, l in half_blks)))
    print()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], args.game)
    xml_path            = require(["smart_tagging"],        args.game)[0]

    # -- Step 1: team metadata --
    print("Loading team metadata from ATD XML …")
    team_of_track, team_names, tag_name_map = load_team_metadata(meta_path)
    print("  Home: {:20s}  Away: {}".format(team_names[HOME], team_names[AWAY]))
    print("  {} track slots ({} home, {} away)".format(
        len(team_of_track),
        int((team_of_track == HOME).sum()),
        int((team_of_track == AWAY).sum())))
    print("  Smart Tagging team map: {}".format(
        {k: team_names[v] for k, v in tag_name_map.items()}))

    # -- Step 2: tracking data --
    print("\nStreaming {} ATD frames …".format(n_frames))
    ball_x, ball_y, nearest, near_dist, mean_speed = read_atd(
        raw_path, team_of_track, n_frames, pitch)
    ball_vis = int(np.sum(~np.isnan(ball_x)))
    print("  Ball visible: {} / {} frames ({:.1f}%)".format(
        ball_vis, n_frames, 100.0 * ball_vis / n_frames))
    print("  Mean player speed (active frames): {:.2f} m/s".format(
        float(mean_speed[mean_speed > 0].mean())))

    # -- Step 3: Smart Tagging --
    print("\nLoading Smart Tagging …")
    tag_dead, restart_team = load_smart_tagging(xml_path, n_frames, tag_name_map, fps)
    n_seeds = int((restart_team != NOBODY).sum())
    print("  Dead-from-tags: {} frames | Restart-seed frames: {}".format(
        int(tag_dead.sum()), n_seeds))

    # -- Step 4: combined dead-ball mask --
    print("\nBuilding dead-ball mask …")
    dead_mask = combined_dead_mask(ball_x, ball_y, mean_speed, tag_dead, n_frames, pitch)
    # Restart seeds override the dead-ball mask (play has restarted)
    dead_mask[restart_team != NOBODY] = False
    print("  spd_dead={:.1f}%  bnd_dead={:.2f}%  tag_dead={:.1f}%  combined={:.1f}%".format(
        100.0 * speed_dead_mask(mean_speed, n_frames).sum() / n_frames,
        100.0 * boundary_dead_mask(ball_x, ball_y, pitch).sum()   / n_frames,
        100.0 * tag_dead.sum()                              / n_frames,
        100.0 * dead_mask.sum()                             / n_frames))

    # -- Step 5: state machine --
    print("\nRunning possession state machine …")
    labels = possession_state_machine(
        nearest, near_dist, dead_mask, restart_team, n_frames, team_names, half_blks)

    # -- Step 6: post-processing --
    print("Enforcing minimum spell length ({} frames = {:.1f} s) …".format(
        MIN_SPELL_FRAMES, MIN_SPELL_FRAMES / fps))
    labels = enforce_min_spell(labels, n_frames, half_blks, MIN_SPELL_FRAMES)

    # -- Stats --
    counts = {lbl: int(np.sum(labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print("\nLabel counts:")
    print("  H={H:>7}  A={A:>7}  D={D:>7}  X={X:>7}".format(**counts))
    alive   = counts["H"] + counts["A"]
    if alive:
        print("  Home share of alive frames: {:.1f}%".format(
            100.0 * counts["H"] / alive))

    # -- Write --
    out_path = os.path.join(args.out, "{}.csv".format(args.name))
    write_submission(out_path, labels)
    print("\nWritten → {}".format(os.path.relpath(out_path, ROOT)))
    print("\nValidate: python evaluation/validate_submission.py {}".format(
        os.path.relpath(out_path, ROOT)))
    print("Score:    python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
