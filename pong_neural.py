"""
Modular 2-player Neural Pong
============================

A Pong game where each paddle can be driven by a Human, a saved ("AI_Model")
brain, or an active ("Training") brain that updates its weights from rewards.

All AI brains are designed as if they always defend the RIGHT side of the
screen. A brain assigned to the LEFT paddle mirrors the X axis before the state
is fed to the network, so the exact same .pkl file plays on either side.

A mouse-driven SETUP MENU lets you pick each side's controller (Human / AI /
Perfect), load a saved brain or start a fresh one, choose whether that side is
learning, and name the file to save to — no need to edit the script or memorize
commands. The constants below just seed the menu's defaults.

In-game controls (also shown on screen)
---------------------------------------
  Left Human  : W / S            Right Human : Up / Down
  1 / 2       : toggle LEFT / RIGHT learning on or off, any time
  S           : save AI brain(s) to their chosen filenames
  M           : back to the setup menu
  Esc / close : quit (plots the accuracy history on exit)

Requires: pygame, numpy, matplotlib
"""

import os
import sys
import pickle
import random

import numpy as np
import pygame

# matplotlib is only needed on exit. Pick the first interactive backend that
# actually imports on this machine; fall back to headless "Agg" (save-only).
# Note: matplotlib.use() doesn't verify a GUI toolkit is installed, so we probe
# the backing module ourselves before committing to it.
import importlib
import matplotlib


def _select_matplotlib_backend():
    candidates = [
        ("MacOSX", None),       # native on macOS, no extra deps
        ("TkAgg", "tkinter"),
        ("QtAgg", "PyQt5"),
    ]
    for backend, probe_module in candidates:
        try:
            if probe_module is not None:
                importlib.import_module(probe_module)
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue
    matplotlib.use("Agg", force=True)   # headless: the PNG still gets written
    return "Agg"


MPL_BACKEND = _select_matplotlib_backend()
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION  (edit these to set up a match)
# ===========================================================================

# Controller per side: "Human" | "AI_Model" | "Training" | "Perfect"
#   "Perfect" is a flawless rule-based bot that always tracks the ball — an
#   ideal practice partner to train a "Training" brain against.
LEFT_CONTROLLER = "Training"
RIGHT_CONTROLLER = "Perfect"

# Brain files used by "AI_Model" (load) and "Training" (save target)
LEFT_MODEL_PATH = "defensive_bot.pkl"     # save target when LEFT is "Training"
RIGHT_MODEL_PATH = "defensive_bot.pkl"    # load source when RIGHT is "AI_Model"

# Window / physics
# WINDOW_WIDTH is the *play field* width (all ball/paddle physics use it).
# A side PANEL is appended to the right of the field for the live training graph.
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PANEL_WIDTH = 320           # right-hand panel showing the live accuracy chart
FPS = 60

PADDLE_WIDTH = 12
PADDLE_HEIGHT = 100
PADDLE_SPEED = 6

BALL_SIZE = 10
BALL_BASE_SPEED = 5.0
BALL_SPEEDUP = 1.05          # multiplier applied to speed on every paddle hit
BALL_MAX_SPEED = 12.0        # used for input normalization and a speed cap

# Training
GENERATION_LENGTH = 20       # number of ball resets (misses) before a generation rolls over
LEARNING_RATE = 0.01         # policy-gradient (REINFORCE) step size
REWARD_HIT = 1.0
REWARD_MISS = -10.0
REWARD_DISCOUNT = 0.9        # credit decays into the past from a hit/miss event

# Imitation pre-training (DISABLED): when on, each frame supervises a Training
# brain toward the analytically-correct move — fast but spoon-fed. Left here so
# it can be toggled back on, but we now learn by pure RL instead.
IMITATION = False
IMITATION_LR = 0.01          # initial supervised step size (lower = slower, watchable climb)
IMITATION_DECAY = 0.85       # multiply IMITATION_LR by this every generation

# Dense reward shaping (pure RL): in addition to the sparse +1/-10 at hit/miss,
# give a small per-frame reward when the sampled action moves the paddle toward
# the ball's predicted intercept (and a small penalty when it moves away). This
# keeps learning purely reward-driven (the net still explores and discovers the
# policy itself) but provides a dense, watchable gradient instead of waiting for
# a rally to end. Keep the scale small so the true +1/-10 outcomes still dominate.
REWARD_SHAPING = True
SHAPING_SCALE = 0.3          # per-frame shaping reward magnitude (try 0.1-0.5)

# Colors
BLACK = (10, 10, 15)
WHITE = (235, 235, 235)
GREY = (90, 90, 100)
GREEN = (80, 220, 120)
BLUE = (90, 160, 255)
PANEL_BG = (18, 18, 26)
GRID = (42, 42, 58)


# ===========================================================================
# NEURAL BRAIN  (numpy-only feed-forward net + REINFORCE policy gradient)
# ===========================================================================

class NeuralBrain:
    """5 -> 8 -> 3 feed-forward policy network.

    Inputs (all normalized to ~[0,1], from the RIGHT-defender perspective):
        ball_x, ball_y, ball_dx, ball_dy, paddle_y
    Outputs (softmax over 3 actions):
        0 = move up, 1 = stay, 2 = move down
    """

    N_INPUTS = 5
    N_HIDDEN = 8
    N_OUTPUTS = 3

    def __init__(self):
        rng = np.random.default_rng()
        # Xavier-ish initialization keeps early activations sane.
        self.W1 = rng.standard_normal((self.N_INPUTS, self.N_HIDDEN)) * np.sqrt(1.0 / self.N_INPUTS)
        self.b1 = np.zeros(self.N_HIDDEN)
        self.W2 = rng.standard_normal((self.N_HIDDEN, self.N_OUTPUTS)) * np.sqrt(1.0 / self.N_HIDDEN)
        self.b2 = np.zeros(self.N_OUTPUTS)

        # Trajectory buffer of per-frame gradients awaiting a reward signal.
        self._pending = []

    # --- inference -------------------------------------------------------
    def _forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()                       # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return h, probs

    def act(self, x, sample=True, remember=False):
        """Return an action (0/1/2).

        `sample` chooses stochastically from the policy (how a policy-gradient
        agent is meant to act) vs. greedy argmax. `remember` stores the gradient
        so a later reward can reinforce the action. These are independent: a
        frozen brain still samples its policy (faithful play) but doesn't
        remember; a learning brain samples AND remembers. Greedy argmax is
        avoided for play because these policies' best behaviour lives in the
        sampling distribution — argmax often collapses to a single action."""
        h, probs = self._forward(x)
        if sample:
            action = int(np.random.choice(self.N_OUTPUTS, p=probs))
        else:
            action = int(np.argmax(probs))
        if remember:
            self._remember(x, h, probs, action)
        return action

    # --- learning --------------------------------------------------------
    def _remember(self, x, h, probs, action):
        # Gradient of log pi(action) w.r.t. each parameter (REINFORCE).
        dlogits = -probs
        dlogits[action] += 1.0                        # (one_hot - probs)

        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh_raw = dh * (1.0 - h * h)                    # tanh'
        dW1 = np.outer(x, dh_raw)
        db1 = dh_raw

        self._pending.append((dW1, db1, dW2, db2))

    def reward(self, value, gamma=REWARD_DISCOUNT):
        """Apply `value` to the frames since the last reward, with *discounted*
        credit assignment: the frame nearest the event (last) gets the full
        reward and earlier frames get gamma**k. This blames/credits the actions
        that actually mattered (the ones just before a hit or miss) instead of
        punishing an entire rally flat — the latter collapses the policy."""
        if not self._pending:
            return
        scale0 = LEARNING_RATE * value
        g = 1.0
        for dW1, db1, dW2, db2 in reversed(self._pending):
            s = scale0 * g
            self.W1 += s * dW1
            self.b1 += s * db1
            self.W2 += s * dW2
            self.b2 += s * db2
            g *= gamma
        self._pending.clear()

    def reward_last(self, value):
        """Apply an immediate reward to only the most recent frame (dense
        shaping), leaving the trajectory buffer intact so the frame still
        receives its share of the terminal +1/-10 when the rally ends."""
        if not self._pending:
            return
        scale = LEARNING_RATE * value
        dW1, db1, dW2, db2 = self._pending[-1]
        self.W1 += scale * dW1
        self.b1 += scale * db1
        self.W2 += scale * dW2
        self.b2 += scale * db2

    def learn_supervised(self, x, target_action, lr):
        """One cross-entropy gradient step pushing the policy toward
        `target_action` (imitation learning). Gradient of CE w.r.t. logits is
        (probs - one_hot(target))."""
        h, probs = self._forward(x)
        dlogits = probs.copy()
        dlogits[target_action] -= 1.0

        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh_raw = dh * (1.0 - h * h)
        dW1 = np.outer(x, dh_raw)
        db1 = dh_raw

        # gradient *descent* to minimize cross-entropy
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    # --- persistence -----------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"W1": self.W1, "b1": self.b1,
                         "W2": self.W2, "b2": self.b2}, f)

    @classmethod
    def load(cls, path):
        brain = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        brain.W1, brain.b1 = data["W1"], data["b1"]
        brain.W2, brain.b2 = data["W2"], data["b2"]
        return brain


# ===========================================================================
# STATE ENCODING  (perspective inversion trick)
# ===========================================================================

def encode_state(ball, paddle_y, side):
    """Build the normalized 5-vector the network expects.

    The network always thinks it defends the RIGHT side. For a LEFT paddle we
    mirror the X axis: ball_x -> W - ball_x and ball_dx -> -ball_dx.
    """
    bx, by = ball.x, ball.y
    bdx, bdy = ball.dx, ball.dy
    if side == "left":
        bx = WINDOW_WIDTH - bx
        bdx = -bdx

    return np.array([
        bx / WINDOW_WIDTH,
        by / WINDOW_HEIGHT,
        (bdx + BALL_MAX_SPEED) / (2 * BALL_MAX_SPEED),
        (bdy + BALL_MAX_SPEED) / (2 * BALL_MAX_SPEED),
        paddle_y / WINDOW_HEIGHT,
    ], dtype=np.float64)


def predicted_intercept_y(ball, target_x):
    """Predict the ball's center-y when it reaches `target_x`, accounting for
    reflections off the top/bottom walls. If the ball is moving away from the
    paddle, just hold on its current y (stay centered, ready)."""
    if ball.dx == 0:
        return ball.y + BALL_SIZE / 2
    t = (target_x - ball.x) / ball.dx
    if t <= 0:                                  # ball moving away from this paddle
        return ball.y + BALL_SIZE / 2
    y = ball.y + ball.dy * t                    # unbounded vertical position
    span = WINDOW_HEIGHT - BALL_SIZE            # fold into a triangle wave
    y = y % (2 * span)
    if y < 0:
        y += 2 * span
    if y > span:
        y = 2 * span - y
    return y + BALL_SIZE / 2


def teacher_action(ball, paddle, target_x):
    """The analytically-correct move (0=up, 1=stay, 2=down) for `paddle`: aim
    its center at the ball's predicted intercept. This is the lesson imitation
    learning supervises against."""
    target = predicted_intercept_y(ball, target_x)
    center = paddle.y + PADDLE_HEIGHT / 2
    if target < center - PADDLE_SPEED:
        return 0
    if target > center + PADDLE_SPEED:
        return 2
    return 1


# ===========================================================================
# CONTROLLERS
# ===========================================================================

class HumanController:
    kind = "Human"

    def __init__(self, side):
        self.side = side
        if side == "left":
            self.up_key, self.down_key = pygame.K_w, pygame.K_s
        else:
            self.up_key, self.down_key = pygame.K_UP, pygame.K_DOWN

    def decide(self, ball, paddle):
        keys = pygame.key.get_pressed()
        if keys[self.up_key]:
            return -1
        if keys[self.down_key]:
            return 1
        return 0


class PerfectController:
    """A flawless rule-based opponent: it snaps its paddle so the center always
    tracks the ball's vertical position, so it never misses. Useful as a stable
    practice partner that keeps the rally alive while the other side learns."""

    kind = "Perfect"

    def __init__(self, side):
        self.side = side

    def decide(self, ball, paddle):
        # Center the paddle on the ball directly (clamped to the field) and
        # return 0 so the game's move() doesn't add anything on top.
        target = (ball.y + BALL_SIZE / 2) - PADDLE_HEIGHT / 2
        paddle.y = max(0, min(WINDOW_HEIGHT - PADDLE_HEIGHT, target))
        return 0


class AIController:
    """Drives a paddle from a NeuralBrain. `learning` can be toggled live: when
    on, the brain explores and updates its weights from rewards; when off it
    plays greedily and is frozen. The same controller covers both the old
    "Training" and "AI_Model" roles — the only difference is the learning flag."""

    kind = "AI"

    def __init__(self, side, brain, learning, save_path, source_name="(new)"):
        self.side = side
        self.brain = brain
        self.learning = learning
        self.save_path = save_path
        self.source_name = source_name        # what was loaded, for display
        self.imit_lr = IMITATION_LR            # decays each generation (imitation only)

    def set_learning(self, on):
        self.learning = on
        if not on:
            self.brain._pending.clear()        # drop any half-finished trajectory

    def decide(self, ball, paddle):
        x = encode_state(ball, paddle.y, self.side)
        # Always sample the policy (faithful play); only remember/learn when on.
        action = self.brain.act(x, sample=True, remember=self.learning)
        if self.learning:
            front_x = paddle.x + PADDLE_WIDTH if self.side == "left" else paddle.x
            if IMITATION and self.imit_lr > 1e-4:
                # imitation: supervise toward the correct move for THIS paddle.
                self.brain.learn_supervised(x, teacher_action(ball, paddle, front_x), self.imit_lr)
            if REWARD_SHAPING:
                # dense RL reward: did the sampled action move toward the intercept?
                target = predicted_intercept_y(ball, front_x)
                center = paddle.y + PADDLE_HEIGHT / 2
                chosen = action - 1                          # -1 up / 0 stay / +1 down
                desired = 0
                if target < center - PADDLE_SPEED:
                    desired = -1
                elif target > center + PADDLE_SPEED:
                    desired = 1
                if desired == 0:
                    agreement = 1.0 if chosen == 0 else -0.5  # reward holding still when aligned
                else:
                    agreement = 1.0 if chosen == desired else (-1.0 if chosen == -desired else 0.0)
                self.brain.reward_last(SHAPING_SCALE * agreement)
        return action - 1            # 0/1/2 -> -1/0/1

    def decay_imitation(self):
        self.imit_lr *= IMITATION_DECAY

    def on_hit(self):
        if self.learning:
            self.brain.reward(REWARD_HIT)

    def on_miss(self):
        if self.learning:
            self.brain.reward(REWARD_MISS)


def build_from_config(side, cfg):
    """Construct a controller for `side` from a menu config dict:
        {"kind": "Human"|"AI"|"Perfect", "brain": filename|None,
         "learning": bool, "save": filename}
    """
    kind = cfg["kind"]
    if kind == "Human":
        return HumanController(side)
    if kind == "Perfect":
        print(f"[{side}] Perfect rule-based opponent (always tracks the ball).")
        return PerfectController(side)
    if kind == "AI":
        source = cfg.get("brain")
        if source and os.path.exists(source):
            brain = NeuralBrain.load(source)
            source_name = source
            print(f"[{side}] Loaded brain '{source}' (learning={'on' if cfg['learning'] else 'off'}).")
        else:
            brain = NeuralBrain()
            source_name = "(new)"
            print(f"[{side}] Fresh random brain (learning={'on' if cfg['learning'] else 'off'}).")
        return AIController(side, brain, learning=cfg["learning"],
                            save_path=cfg["save"], source_name=source_name)
    raise ValueError(f"Unknown controller kind: {kind!r}")


# ===========================================================================
# GAME OBJECTS
# ===========================================================================

class Ball:
    def __init__(self):
        self.reset(serve_to=random.choice(["left", "right"]))

    def reset(self, serve_to):
        self.x = WINDOW_WIDTH / 2
        self.y = WINDOW_HEIGHT / 2
        angle = random.uniform(-0.78, 0.78)          # ~ +/- 45 degrees
        direction = -1 if serve_to == "left" else 1
        self.dx = direction * BALL_BASE_SPEED * np.cos(angle)
        self.dy = BALL_BASE_SPEED * np.sin(angle)

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), BALL_SIZE, BALL_SIZE)

    def speed(self):
        return (self.dx ** 2 + self.dy ** 2) ** 0.5


class Paddle:
    def __init__(self, side):
        self.side = side
        self.x = 20 if side == "left" else WINDOW_WIDTH - 20 - PADDLE_WIDTH
        self.y = (WINDOW_HEIGHT - PADDLE_HEIGHT) / 2

    def move(self, direction):
        self.y += direction * PADDLE_SPEED
        self.y = max(0, min(WINDOW_HEIGHT - PADDLE_HEIGHT, self.y))

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_WIDTH, PADDLE_HEIGHT)


class SideStats:
    """Per-paddle hit/miss tracking and per-generation accuracy history."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.history = []        # accuracy per completed generation

    def accuracy(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def roll_generation(self):
        self.history.append(self.accuracy())
        self.hits = 0
        self.misses = 0


# ===========================================================================
# MAIN GAME
# ===========================================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH + PANEL_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Pong")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 28, bold=True)

    # Seed the menu from the top-of-file config constants.
    config = default_menu_config()

    while True:
        config = run_menu(screen, clock, font, big_font, config)
        if config is None:                       # window closed / quit from menu
            break
        result, stats, controllers = run_game(screen, clock, font, big_font, config)
        if result == "quit":
            plot_history(stats, *controllers)    # blocking graph as we exit
            break
        # result == "menu": loop back, keeping the last config as defaults

    pygame.quit()


# --- menu -------------------------------------------------------------------

def _list_brains():
    return sorted(f for f in os.listdir(".") if f.endswith(".pkl"))


def default_menu_config():
    """Translate the LEFT/RIGHT constants into the menu's per-side dicts."""
    def one(side, kind_const, path_const):
        if kind_const == "Human":
            kind, learning = "Human", False
        elif kind_const == "Perfect":
            kind, learning = "Perfect", False
        elif kind_const == "AI_Model":
            kind, learning = "AI", False
        else:                                    # "Training"
            kind, learning = "AI", True
        brain = path_const if (kind_const == "AI_Model" and os.path.exists(path_const)) else None
        save = path_const if path_const else f"brain_{side}.pkl"
        return {"kind": kind, "brain": brain, "learning": learning, "save": save}
    return {"left": one("left", LEFT_CONTROLLER, LEFT_MODEL_PATH),
            "right": one("right", RIGHT_CONTROLLER, RIGHT_MODEL_PATH)}


def _button(screen, font, text, rect, active=False, enabled=True, color=None):
    rect = pygame.Rect(rect)
    if color is None:
        color = (45, 90, 150) if active else (38, 38, 52)
    if not enabled:
        color = (26, 26, 34)
    pygame.draw.rect(screen, color, rect, border_radius=6)
    pygame.draw.rect(screen, GREY, rect, 1, border_radius=6)
    label = font.render(text, True, WHITE if enabled else GREY)
    screen.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                        rect.y + (rect.h - label.get_height()) // 2))
    return rect


def run_menu(screen, clock, font, big_font, config):
    """Interactive setup screen. Returns the chosen config dict, or None if the
    user closed the window."""
    brains = ["(new random brain)"] + _list_brains()

    def brain_index(side):
        b = config[side]["brain"]
        return brains.index(b) if (b in brains) else 0

    bidx = {"left": brain_index("left"), "right": brain_index("right")}
    editing = {"side": None}                     # which save-name field has focus

    while True:
        regions = []                             # (rect, callback) for mouse hits
        screen.fill(BLACK)

        title = big_font.render("NEURAL PONG  —  SETUP", True, WHITE)
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 28))

        for side, x0 in (("left", 70), ("right", 600)):
            cfg = config[side]
            col = BLUE if side == "left" else GREEN
            head = big_font.render(side.upper(), True, col)
            screen.blit(head, (x0, 90))

            # --- controller type ---
            screen.blit(font.render("Controller:", True, GREY), (x0, 140))
            for i, k in enumerate(("Human", "AI", "Perfect")):
                r = _button(screen, font, k, (x0 + i * 150, 166, 140, 34),
                            active=(cfg["kind"] == k))
                regions.append((r, lambda s=side, k=k: config[s].update(kind=k)))

            if cfg["kind"] == "AI":
                # --- brain source (cycle through files) ---
                screen.blit(font.render("Brain:", True, GREY), (x0, 220))
                rl = _button(screen, font, "<", (x0, 246, 34, 34))
                rr = _button(screen, font, ">", (x0 + 416, 246, 34, 34))
                name = brains[bidx[side]]
                box = pygame.Rect(x0 + 40, 246, 372, 34)
                pygame.draw.rect(screen, (30, 30, 42), box, border_radius=6)
                pygame.draw.rect(screen, GREY, box, 1, border_radius=6)
                nm = font.render(name, True, WHITE)
                screen.blit(nm, (box.x + 10, box.y + 7))

                def cycle(side, delta):
                    bidx[side] = (bidx[side] + delta) % len(brains)
                    sel = brains[bidx[side]]
                    config[side]["brain"] = None if bidx[side] == 0 else sel
                    if bidx[side] != 0:
                        config[side]["save"] = sel        # default to save back to source
                regions.append((rl, lambda s=side: cycle(s, -1)))
                regions.append((rr, lambda s=side: cycle(s, +1)))

                # --- learning toggle ---
                screen.blit(font.render("Learning:", True, GREY), (x0, 300))
                on = cfg["learning"]
                r = _button(screen, font, "ON" if on else "OFF",
                            (x0, 326, 140, 34), active=on,
                            color=(40, 120, 70) if on else (120, 50, 50))
                regions.append((r, lambda s=side: config[s].update(learning=not config[s]["learning"])))

                # --- save-as field ---
                screen.blit(font.render("Save as:", True, GREY), (x0, 380))
                fld = pygame.Rect(x0, 406, 450, 34)
                focused = editing["side"] == side
                pygame.draw.rect(screen, (30, 30, 42), fld, border_radius=6)
                pygame.draw.rect(screen, WHITE if focused else GREY, fld, 2 if focused else 1,
                                 border_radius=6)
                caret = "|" if (focused and pygame.time.get_ticks() // 400 % 2) else ""
                screen.blit(font.render(cfg["save"] + caret, True, WHITE), (fld.x + 10, fld.y + 7))
                regions.append((fld, lambda s=side: editing.update(side=s)))

        # --- start + hint ---
        start = _button(screen, big_font, "START  ▶", (screen.get_width() // 2 - 110, 500, 220, 48),
                        color=(40, 120, 70))
        regions.append((start, lambda: "start"))
        hint = font.render("Click to configure each side.  In game: 1/2 toggle learning · "
                           "S save · M menu · Esc quit", True, GREY)
        screen.blit(hint, (screen.get_width() // 2 - hint.get_width() // 2, 560))

        pygame.display.flip()
        clock.tick(FPS)

        # --- events ------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                editing["side"] = None
                for rect, cb in regions:
                    if rect.collidepoint(event.pos):
                        if cb() == "start":
                            return config
                        break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if editing["side"]:
                    s = editing["side"]
                    if event.key == pygame.K_RETURN:
                        editing["side"] = None
                    elif event.key == pygame.K_BACKSPACE:
                        config[s]["save"] = config[s]["save"][:-1]
                    elif event.unicode and event.unicode.isprintable() and len(config[s]["save"]) < 32:
                        config[s]["save"] += event.unicode


# --- game session -----------------------------------------------------------

def run_game(screen, clock, font, big_font, config):
    """Run one match. Returns (result, stats, (left_ctrl, right_ctrl)) where
    result is "menu" (return to setup) or "quit" (exit the program)."""
    left_ctrl = build_from_config("left", config["left"])
    right_ctrl = build_from_config("right", config["right"])

    left_paddle = Paddle("left")
    right_paddle = Paddle("right")
    ball = Ball()

    stats = {"left": SideStats(), "right": SideStats()}
    generation = 1
    resets_this_gen = 0

    def handle_miss(conceding_side, conceding_ctrl):
        nonlocal generation, resets_this_gen
        if hasattr(conceding_ctrl, "on_miss"):
            conceding_ctrl.on_miss()
        stats[conceding_side].misses += 1
        ball.reset(serve_to=conceding_side)
        resets_this_gen += 1
        if resets_this_gen >= GENERATION_LENGTH:
            for side in ("left", "right"):
                stats[side].roll_generation()
            for ctrl in (left_ctrl, right_ctrl):
                if isinstance(ctrl, AIController) and ctrl.learning:
                    ctrl.decay_imitation()
            print(f"--- Generation {generation} complete | "
                  f"left acc {stats['left'].history[-1]:.2f} | "
                  f"right acc {stats['right'].history[-1]:.2f} ---")
            generation += 1
            resets_this_gen = 0

    def save_brains():
        saved = False
        for ctrl in (left_ctrl, right_ctrl):
            if isinstance(ctrl, AIController):
                ctrl.brain.save(ctrl.save_path)
                print(f"Saved {ctrl.side} brain to {ctrl.save_path} "
                      f"(gen {generation}, acc {stats[ctrl.side].accuracy():.2f})")
                saved = True
        if not saved:
            print("No AI side active — nothing to save.")

    def toggle_learning(side):
        ctrl = left_ctrl if side == "left" else right_ctrl
        if isinstance(ctrl, AIController):
            ctrl.set_learning(not ctrl.learning)
            print(f"{side} learning -> {'ON' if ctrl.learning else 'OFF'}")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", stats, (left_ctrl, right_ctrl)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit", stats, (left_ctrl, right_ctrl)
                elif event.key == pygame.K_m:
                    return "menu", stats, (left_ctrl, right_ctrl)
                elif event.key == pygame.K_s:
                    save_brains()
                elif event.key == pygame.K_1:
                    toggle_learning("left")
                elif event.key == pygame.K_2:
                    toggle_learning("right")

        left_paddle.move(left_ctrl.decide(ball, left_paddle))
        right_paddle.move(right_ctrl.decide(ball, right_paddle))

        ball.x += ball.dx
        ball.y += ball.dy

        if ball.y <= 0:
            ball.y = 0
            ball.dy = abs(ball.dy)
        elif ball.y >= WINDOW_HEIGHT - BALL_SIZE:
            ball.y = WINDOW_HEIGHT - BALL_SIZE
            ball.dy = -abs(ball.dy)

        if ball.dx < 0 and ball.rect.colliderect(left_paddle.rect):
            ball.x = left_paddle.rect.right
            ball.dx = abs(ball.dx)
            _apply_spin(ball, left_paddle)
            _speed_up(ball)
            stats["left"].hits += 1
            if hasattr(left_ctrl, "on_hit"):
                left_ctrl.on_hit()
        elif ball.dx > 0 and ball.rect.colliderect(right_paddle.rect):
            ball.x = right_paddle.rect.left - BALL_SIZE
            ball.dx = -abs(ball.dx)
            _apply_spin(ball, right_paddle)
            _speed_up(ball)
            stats["right"].hits += 1
            if hasattr(right_ctrl, "on_hit"):
                right_ctrl.on_hit()

        if ball.x < -BALL_SIZE:
            handle_miss("left", left_ctrl)
        elif ball.x > WINDOW_WIDTH:
            handle_miss("right", right_ctrl)

        _draw(screen, font, big_font, ball, left_paddle, right_paddle,
              left_ctrl, right_ctrl, stats, generation, resets_this_gen)
        _draw_graph(screen, font, stats, (left_ctrl, right_ctrl), generation)
        pygame.display.flip()
        clock.tick(FPS)


def _apply_spin(ball, paddle):
    """Add a little vertical influence based on where the ball struck the paddle."""
    offset = (ball.y + BALL_SIZE / 2) - (paddle.y + PADDLE_HEIGHT / 2)
    ball.dy += (offset / (PADDLE_HEIGHT / 2)) * 1.5


def _speed_up(ball):
    speed = ball.speed()
    target = min(speed * BALL_SPEEDUP, BALL_MAX_SPEED)
    if speed > 0:
        factor = target / speed
        ball.dx *= factor
        ball.dy *= factor


def _draw(screen, font, big_font, ball, lp, rp, lc, rc, stats, generation, resets):
    screen.fill(BLACK)

    # center net
    for y in range(0, WINDOW_HEIGHT, 30):
        pygame.draw.rect(screen, GREY, (WINDOW_WIDTH // 2 - 2, y, 4, 18))

    pygame.draw.rect(screen, BLUE, lp.rect)
    pygame.draw.rect(screen, GREEN, rp.rect)
    pygame.draw.ellipse(screen, WHITE, ball.rect)

    # HUD
    header = big_font.render(f"Generation {generation}   ({resets}/{GENERATION_LENGTH})",
                             True, WHITE)
    screen.blit(header, (WINDOW_WIDTH // 2 - header.get_width() // 2, 12))

    def label(side, ctrl):
        if isinstance(ctrl, AIController):
            tag = f"AI {'⚡learning' if ctrl.learning else 'frozen'}"
        else:
            tag = ctrl.kind
        s = stats[side]
        return (f"{side.upper()} [{tag}]  hits {s.hits}  miss {s.misses}"
                f"  acc {s.accuracy():.2f}")

    screen.blit(font.render(label("left", lc), True, BLUE), (20, WINDOW_HEIGHT - 56))
    screen.blit(font.render(label("right", rc), True, GREEN), (20, WINDOW_HEIGHT - 30))

    hint = font.render("1/2:learn  S:save  M:menu  Esc:quit", True, GREY)
    screen.blit(hint, (WINDOW_WIDTH - hint.get_width() - 20, WINDOW_HEIGHT - 30))


def _draw_graph(screen, font, stats, controllers, generation):
    """Live accuracy-per-generation chart drawn in the right-hand panel.

    Plots each Training side's completed-generation history plus a provisional
    point for the in-progress generation, so progress is visible immediately
    rather than only at generation boundaries.
    """
    px = WINDOW_WIDTH                       # panel origin x
    pygame.draw.rect(screen, PANEL_BG, (px, 0, PANEL_WIDTH, WINDOW_HEIGHT))
    pygame.draw.line(screen, WHITE, (px, 0), (px, WINDOW_HEIGHT), 2)

    title = font.render("Training Accuracy / Gen", True, WHITE)
    screen.blit(title, (px + 16, 16))

    # plot area
    left = px + 52
    right = px + PANEL_WIDTH - 18
    top = 52
    bottom = WINDOW_HEIGHT - 46
    pw = right - left
    ph = bottom - top
    pygame.draw.rect(screen, GREY, (left, top, pw, ph), 1)

    def ymap(acc):
        return bottom - acc * ph

    # horizontal gridlines + y labels (0.0 .. 1.0)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = ymap(frac)
        pygame.draw.line(screen, GRID, (left, yy), (right, yy), 1)
        lbl = font.render(f"{frac:.2f}", True, GREY)
        screen.blit(lbl, (px + 12, yy - 10))

    # build series for any AI side (history + provisional running point)
    series = []
    for ctrl in controllers:
        if isinstance(ctrl, AIController):
            side = ctrl.side
            hist = list(stats[side].history)
            running = stats[side].accuracy()
            color = BLUE if side == "left" else GREEN
            series.append((side, hist, running, color))

    if not series:
        note = font.render("(no AI side)", True, GREY)
        screen.blit(note, (left, top + ph // 2))
        return

    max_pts = max(len(h) + 1 for _, h, _, _ in series)
    max_pts = max(max_pts, 2)

    def xmap(i):
        return left + (i / (max_pts - 1)) * pw

    for side, hist, running, color in series:
        pts = [(xmap(i), ymap(a)) for i, a in enumerate(hist)]
        pts.append((xmap(len(hist)), ymap(running)))   # in-progress point
        if len(pts) >= 2:
            pygame.draw.lines(screen, color, False, pts, 2)
        for x, y in pts[:-1]:
            pygame.draw.circle(screen, color, (int(x), int(y)), 3)
        # hollow marker on the live (in-progress) point
        lx, ly = pts[-1]
        pygame.draw.circle(screen, WHITE, (int(lx), int(ly)), 4, 1)

    # x-axis label + per-side legend with current values
    xlbl = font.render(f"generation (now {generation})", True, GREY)
    screen.blit(xlbl, (left, bottom + 12))
    ly = top + ph + 2
    for i, (side, hist, running, color) in enumerate(series):
        leg = font.render(f"{side}: {running:.2f}", True, color)
        screen.blit(leg, (right - leg.get_width() - 4 - i * 0, bottom - 18 - i * 20))


def plot_history(stats, left_ctrl, right_ctrl):
    """Plot per-generation accuracy for any side that produced history, save a
    PNG next to its brain, and show the figure."""
    series = []
    for ctrl in (left_ctrl, right_ctrl):
        side = ctrl.side
        if stats[side].history:
            series.append((side, ctrl, stats[side].history))

    if not series:
        print("No completed generations — skipping performance graph.")
        return

    plt.figure(figsize=(8, 5))
    for side, ctrl, history in series:
        gens = range(1, len(history) + 1)
        plt.plot(gens, history, marker="o", label=f"{side} [{ctrl.kind}]")

    plt.title("Pong AI Accuracy per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy  (hits / (hits + misses))")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save next to a brain file if one is being trained, else a default name.
    save_base = "pong"
    for _, ctrl, _ in series:
        if getattr(ctrl, "save_path", None):
            save_base = os.path.splitext(ctrl.save_path)[0]
            break
    out = f"{save_base}_history.png"
    plt.savefig(out, dpi=120)
    print(f"Saved performance graph to {out}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
