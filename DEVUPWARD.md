# More Than A Hack 2027 — Technical Report
**Team Name:** DEVUPWARD  
**Challenge:** Phase 1: Ball Possession Inference  
**Submission Repository:** [github.com/zoubaax/more-than-a-hack-2027](https://github.com/zoubaax/more-than-a-hack-2027)  
**Deliverable File:** `DEVUPWARD.csv` (199,319 frames)

---

## 1. Executive Summary & Headline Results

For every video frame of FC Barcelona vs Athletic Club (199,319 frames at 25 fps, ~158,857 graded frames across both halves), our system produces an algorithm-driven classification label in `{"H", "A", "D", "X"}`.

Rather than treating possession as an isolated per-frame nearest-neighbour classification (which generates extreme temporal jitter and collapses the ranking metric `spellF1`), we designed a **Physics- and Kinematics-Driven Possession State Machine** combined with a **Collective Stoppage Engine**.

### Benchmark Comparison (Public Ground Truth)

| Method | macroF1 | spellF1 | Possessions (GT=17) | Mean Duration (GT=14.2s) | Latency | Share Error |
|---|---|---|---|---|---|---|
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 s | -3.1 f | 0.046 |
| `baseline_smooth50` *(floor)* | 0.6207 | 0.3256 | 26 | 10.8 s | +4.0 f | 0.048 |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 s | -7.7 f | 0.052 |
| **DEVUPWARD (Ours)** | **0.8050** | **0.5714** | **18** | **14.6 s (+3%)** | **+0.8 f** | **0.009 (<1%)** |

* **`spellF1`:** 0.5714 (+135% improvement over the published full-match baseline floor of 0.2433).
* **`macroF1`:** 0.8050 (+0.18 over the best baseline).
* **Dead-Ball F1:** 0.7138 with 90.9% precision (up from 0.3225).
* **Possession Structure:** Recovers 18 possession spells of mean duration 14.6s against 17 true spells of mean 14.2s.
* **Transition Alignment:** Near-zero average transition latency (+0.8 frames, < 32 ms).

---

## 2. Technical Approach & Architecture

Our architecture is structured into four sequential, data-driven stages:

```
Automatic Tracking Data (ATD) + Smart Tagging XML
                     │
                     ▼
  [Stage 1: Multi-Signal Dead-Ball & Stoppage Engine]
  • Collective player speed smoothing (< 1.05 m/s drop)
  • Set-piece pre-restart dead-ball window (Smart Tagging)
  • Physical pitch boundary exit detection (x, y outside pitch + 1m)
                     │
                     ▼
  [Stage 2: Active-Play Occlusion Bridging]
  • Hold current possession across unobserved ball frames (passes, headers)
  • Prevent spurious dead-ball splits during live play
                     │
                     ▼
  [Stage 3: Hysteresis Turnover State Machine]
  • Controlled possession handover requires N_CTRL_FRAMES (8 frames)
  • Eliminates false transitions from single-frame touches or deflections
                     │
                     ▼
  [Stage 4: Spell Absorption & Boundary Enforcement]
  • Enforce minimum spell duration (50 frames / 2.0s)
  • Half-boundary masking (Frames outside graded halves marked 'X')
```

### Why this approach over heavy deep learning / vision models?
1. **Computational Feasibility & Submission Constraints:** Processing 199,319 1080p video frames via large spatio-temporal video architectures (e.g. Video Vision Transformers, SlowFast) requires days of distributed GPU compute. In contrast, Metrica's FIFA/EPTS automatic tracking data already captures calibrated 2D pitch coordinates.
2. **Physical Interpretability & Generalizability:** The physics of collective player speed and possession continuity generalize across football matches without overfitting visual artifacts of camera lighting or jersey colors.
3. **Execution Speed:** Our entire pipeline processes the full 2-hour match in **under 4 seconds** on standard CPU with $O(n)$ time complexity.

---

## 3. Data Assets Used & Dropped

| Asset | Role in Pipeline | Status | Rationale |
|---|---|---|---|
| **Tactical ATD Raw (`BAR-ATH_tactical_atd.txt`)** | Core Input | **Used** | Provides ball coordinates, player pitch positions, and speeds for all 199,319 frames. |
| **Tactical ATD Meta (`BAR-ATH_tactical_atd.xml`)** | Core Metadata | **Used** | Dynamically resolves team identities and track-slot mappings without hardcoded strings. |
| **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`)** | Restart Prior | **Used** | Identifies the 59 set pieces (corners, throw-ins, goal kicks, kick-offs) to mark pre-restart dead periods. |
| **Homography (`BAR-ATH_tactical_homography.csv`)** | Diagnostics | **Used for Validation** | Used to project pitch coordinates onto image space for visual error verification. |
| **Tactical Video (`BAR-ATH_tactical.mp4`)** | Visual Spot-Checking | **Used for Debugging** | Used to manually inspect transition error timestamps reported in evaluation transition logs. |
| **Broadcast Assets (Video, ATD, Homography)** | Secondary Stream | **Dropped** | Broadcast feed contains cuts, replays, and is not synchronized to the tactical submission timeline. Merging timelines would introduce frame alignment error without providing additional spatial coverage. |

---

## 4. Training Data & External Annotations

* **No External or Leaked Data:** In strict adherence to Rule 2, **no external annotated datasets, no proprietary event feeds (Opta, StatsBomb, Wyscout), and no manual labels** were used.
* **Public 5-Minute Window Role:** The 5-minute ground truth provided in `evaluation/gt/` was strictly used to sanity-check pipeline alignment and metric definitions. Hyperparameters were selected based on the physical distributions of football rather than overfitting to the 17 public possessions.

---

## 5. Possession Rules Implemented

Our state machine implements the following operational definitions:

1. **Dead Ball (`D`):**
   * Ball is marked dead (`D`) if:
     - The smoothed collective speed of all active players drops below `1.05 m/s` (detecting referee whistles, stoppages, and fouls).
     - The ball is tracked outside the pitch boundaries plus a `1.0 m` margin ($x < -1\text{m}$, $x > 106.3\text{m}$, $y < -1\text{m}$, $y > 69.0\text{m}$).
     - The frame falls within the pre-restart interval of a Smart Tagging set piece (`TAG_MARGIN_BEFORE = 50 frames`).
2. **Controlled Turnover vs. Deflection:**
   * A change of possession requires **controlled possession** by the opposing team.
   * If Barcelona (`H`) has the ball and an Athletic (`A`) player makes brief contact (1–3 frames), our hysteresis counter treats it as a deflection or contested challenge; possession remains with Barcelona.
   * A handover is only triggered when opposing players are tracked closest to the ball for at least `N_CTRL_FRAMES = 8` consecutive frames (~320 ms).
3. **Occlusions & Loose Balls:**
   * When the ball coordinate drops out (`NaN`) during active play, possession is maintained with the last possessing team.
   * Spurious microscopic transitions shorter than `MIN_SPELL_FRAMES = 50` (2.0s) are merged into their predecessor.
4. **Halftime and Out-of-Match Windows:**
   * All frames before kickoff, during halftime, and after the final whistle are marked `X`. Under competition rules, these cost zero.

---

## 6. Limitations & Future Directions

1. **Throw-In Team Identification:** In Metrica's Smart Tagging export, all 36 throw-ins carry `Team: N/A`. While collective speed correctly flags the dead-ball window, knowing which team threw the ball in advance would allow instant possession seeding at the first frame of restart.
2. **Aerial Duels:** High balls in crowded boxes often suffer from joint tracking dropouts. Explicit 3D ball trajectory fitting or homography-assisted height estimation could further reduce handover lag on crosses.

---

## 7. Reproduction Instructions

The submission file can be regenerated from a fresh clone in one step:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the submission file (takes ~4 seconds)
python solution/pipeline.py --name DEVUPWARD

# 3. Validate against competition checks
python evaluation/validate_submission.py evaluation/results/DEVUPWARD.csv

# 4. Score against public ground truth
python evaluation/run_evaluation.py
```
Output submission is written to: `evaluation/results/DEVUPWARD.csv`
