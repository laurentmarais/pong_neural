# Neural Pong

A 2-player Pong game in Pygame where each paddle can be driven by a **Human**, a
**neural-network brain** (loaded from a file or learning live), or a flawless
rule-based **Perfect** bot. Brains learn to play by reinforcement learning and
can be saved, reloaded, and pitted against each other.

---

## Requirements

- Python 3.9+
- `pygame`, `numpy`, `matplotlib`

## Setup

```bash
# from the project root
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Run

```bash
./.venv/bin/python pong_neural.py
```

This opens the **setup menu** first — you don't need to edit the script or
memorize any commands.

---

## Setup menu (mouse-driven)

For each side (LEFT / RIGHT) you can configure:

| Control       | What it does                                                            |
|---------------|-------------------------------------------------------------------------|
| **Controller**| `Human`, `AI` (neural net), or `Perfect` (always tracks the ball)       |
| **Brain**     | `<` / `>` cycles through `(new random brain)` and every `.pkl` in the folder — pick one to **load a saved brain**, or start fresh |
| **Learning**  | `ON` / `OFF` — whether this AI updates its weights from rewards          |
| **Save as**   | Click the field and type the filename this brain saves to               |

Click **START ▶** to begin. The top-of-file constants in `pong_neural.py` just
seed the menu's defaults.

## In-game controls

Shown along the bottom of the screen, so nothing needs memorizing:

| Key            | Action                                                   |
|----------------|----------------------------------------------------------|
| **W / S**      | Move the LEFT human paddle up / down                     |
| **↑ / ↓**      | Move the RIGHT human paddle up / down                    |
| **1 / 2**      | Toggle LEFT / RIGHT learning on or off, any time         |
| **S**          | Save the AI brain(s) to their chosen filenames           |
| **M**          | Return to the setup menu                                 |
| **Esc / close**| Quit (saves an accuracy graph PNG and shows it)          |

> **Tip:** returning to the menu (`M`) starts a fresh match and resets the live
> stats/graph. **Press `S` to save a brain you've been training before you leave**,
> otherwise the in-memory weights are lost.

The right-hand panel shows a **live accuracy chart** (hits ÷ (hits + misses) per
generation) so you can watch learning progress in real time.

---

## How the neural net works

### Architecture
A tiny numpy-only feed-forward policy network — **5 inputs → 8 hidden (tanh) → 3
outputs (softmax)**. The whole brain is four weight arrays pickled to a single
`.pkl` file.

**Inputs** (normalized to ~[0, 1]): `ball_x, ball_y, ball_dx, ball_dy, paddle_y`.

**Outputs:** a probability over 3 actions — `move up`, `stay`, `move down`. The
agent **samples** from this distribution (it's a stochastic policy), both while
learning and while playing a frozen brain.

### The perspective-inversion trick
Every brain is designed as if it always defends the **RIGHT** side. When a brain
plays the LEFT paddle, the game mirrors the X axis before feeding the state to
the net (`ball_x → W − ball_x`, `ball_dx → −ball_dx`). This means **the same
`.pkl` file plays correctly on either side**.

### How it learns (reinforcement learning)
Learning is pure RL via **REINFORCE** (policy gradient):

- **Sparse rewards:** `+1` for hitting the ball, `−10` for missing it.
- **Discounted credit assignment:** when a hit/miss happens, the reward is
  applied to the recent frames with the nearest frames credited/blamed most
  (`REWARD_DISCOUNT = 0.9`). This focuses learning on the moves that actually
  mattered, instead of punishing a whole rally flat (which collapses the policy).
- **Dense reward shaping:** each frame also gives a small reward when the
  sampled move heads toward the ball's predicted intercept (and a small penalty
  when it moves away). This turns a slow, sparse signal into a steady,
  watchable one without telling the net the exact answer.

A brain typically climbs from random flailing to a decent defender over **one to
a couple of minutes** of training, with visible ups and downs — that's genuine
trial-and-error, not a scripted ramp.

### Training tips
- Train an AI side against the **Perfect** bot — it never misses, so it keeps
  the rally alive and gives a consistent stream of practice.
- **Longer training = stronger brain.** A brain saved after only a few
  generations will be weak; let the accuracy line climb and *hold high* for a
  while before saving.
- To watch two brains co-evolve, set **both** sides to `AI` with learning `ON`.
- To use a trained brain as a fixed opponent, load it with learning `OFF` — it
  still plays its learned policy, it just stops changing.

### Generations & the graph
A "generation" ends after `GENERATION_LENGTH` misses (default 20). Each
generation's accuracy is recorded; on exit a `<save_name>_history.png` graph of
accuracy-per-generation is written next to the brain and displayed.

---

## Tuning (top of `pong_neural.py`)

| Constant            | Meaning                                                      |
|---------------------|--------------------------------------------------------------|
| `LEARNING_RATE`     | Policy-gradient step size                                    |
| `REWARD_HIT` / `REWARD_MISS` | Sparse reward / penalty                             |
| `REWARD_DISCOUNT`   | How fast credit decays into the past from a hit/miss         |
| `REWARD_SHAPING` / `SHAPING_SCALE` | Enable / strength of the dense per-frame reward |
| `GENERATION_LENGTH` | Misses per generation (smaller = denser graph)              |
| `IMITATION` / `IMITATION_LR` | Optional fast imitation pre-training (off by default) |
| Window / physics    | `WINDOW_WIDTH`, `PADDLE_SPEED`, `BALL_BASE_SPEED`, etc.      |

## Files

- `pong_neural.py` — the whole game.
- `*.pkl` — saved brains.
- `*_history.png` — accuracy graphs written on exit.
