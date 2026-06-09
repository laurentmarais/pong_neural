Role: Expert Python & AI Developer
Task: Build a modular 2-player Pong game in Pygame where paddles can be controlled by Humans, loaded AI brains, or an active Trainable AI Brain. The user can save specific brains with custom names and pit them against each other.

Technical Requirements:
1. Libraries: `pygame`, `numpy`, `pickle`, `matplotlib`

2. Modular Controller Design:
    - Create a paddle controller system that can handle three types of inputs:
        a) "Human" (uses W/S keys for Left, Up/Down keys for Right).
        b) "Static_AI" (loads a saved, non-changing .pkl brain file).
        c) "Training_AI" (an active network updating its weights based on rewards).

3. Perspective Inversion Trick (CRITICAL):
    - The AI network should always be designed as if it is defending the RIGHT side of the screen.
    - If an AI is assigned to the LEFT paddle, the game must mirror the X-coordinates (e.g., ball_x becomes window_width - ball_x, and ball_dx becomes -ball_dx) before feeding the state to the network. This allows the exact same brain file to play on either side seamlessly.

4. Game Modes (Set via variables at the top of the script):
    - Provide a simple configuration section at the top of the script to set:
        - LEFT_CONTROLLER = "Human" / "AI_Model" / "Training"
        - RIGHT_CONTROLLER = "Human" / "AI_Model" / "Training"
        - LEFT_MODEL_PATH = "brain_v1.pkl" (if applicable)
        - RIGHT_MODEL_PATH = "brain_v2.pkl" (if applicable)

5. Training & Reward Conditions:
    - If a paddle is in "Training" mode, it receives a +1 reward for hitting the ball and a -10 penalty for missing it. 
    - The game does not stop when one side misses; the ball resets in the center, and the game tracks total hits per "generation" or "match".

6. Naming & Saving:
    - Pressing the 'S' key prompts the console (or automatically saves) the "Training" network's weights to a unique, custom filename specified in the script configuration (e.g., `save_name = 'defensive_bot.pkl'`).
    - Also save historical accuracy data to plot a performance graph upon exit.

Questions:

1. Neural network architecture — what should the brain's structure be (inputs, hidden layers, outputs)?
   *Example answer:* A small feed-forward net. Inputs (5): ball_x, ball_y, ball_dx, ball_dy, paddle_y — all normalized to [0, 1]. One hidden layer of 8 neurons with ReLU/tanh. Output: 3 nodes (move up / stay / move down) via argmax, or 1 node giving a continuous target paddle position.

2. Learning algorithm — how does the "Training" brain actually update its weights from the +1 / -10 rewards?
   *Example answer:* Use a simple REINFORCE-style policy gradient: record (state, action) pairs each point, apply the reward at the end of the rally, and nudge weights toward actions that preceded rewards. Keep it dependency-free (numpy only, no PyTorch/TensorFlow). A hill-climbing / random-mutation evolutionary update is an acceptable simpler alternative.

3. Config naming — Section 2 uses "Human"/"Static_AI"/"Training_AI" but Section 4 uses "Human"/"AI_Model"/"Training". Which strings should the controller config accept?
   *Example answer:* Standardize on Section 4: LEFT_CONTROLLER / RIGHT_CONTROLLER = "Human" | "AI_Model" | "Training". Treat the Section 2 names as descriptive labels only.

4. Generation / match length — when does a "generation" end and stats roll over?
   *Example answer:* A generation ends after a fixed number of ball resets (e.g. 20 misses total) or after N seconds. At that boundary, log the hit count, append to the accuracy history, and start the next generation. Two "Training" paddles at once is allowed.

5. Game constants — what window size, paddle speed, and ball speed should I use?
   *Example answer:* 800x600 window, paddle height 100px moving at 6 px/frame, ball speed ~5 px/frame with a slight speed-up on each paddle hit, 60 FPS.

6. Ball reset behavior — after a miss, where and which direction does the ball serve?
   *Example answer:* Reset to screen center with a randomized vertical angle and a horizontal direction toward the side that just lost (or alternating), at the base speed.

7. "Accuracy" metric for the exit graph — what exactly is plotted?
   *Example answer:* Plot hits / (hits + misses) per generation as a line over generation number using matplotlib, saved alongside the .pkl (e.g. defensive_bot_history.png) and shown on exit.

8. Saving with 'S' — auto-save to the configured name, or prompt for a filename?
   *Example answer:* Auto-save the active Training brain to the configured save_name (e.g. 'defensive_bot.pkl') without prompting, printing a confirmation to the console.

Answer:

1. Architecture: feed-forward net, 5 inputs → 8 hidden (tanh) → 3 outputs (up / stay / down), action = argmax.
   Inputs normalized to [0,1]: ball_x/W, ball_y/H, (ball_dx+max)/(2*max), (ball_dy+max)/(2*max), paddle_y/H.
   Weights stored as plain numpy arrays so the whole brain pickles to one .pkl file. Always framed as defending the RIGHT side; the left paddle mirrors X before feeding the state (per the Perspective Inversion Trick).

2. Learning: REINFORCE-style policy gradient, numpy-only (no torch/TF).
   - Softmax the 3 outputs into action probabilities; sample an action and store (state, action, prob) each frame of the rally.
   - On a hit apply reward +1, on a miss apply -10, to the actions taken since the last reward.
   - Update: w += lr * reward * grad(log π(action)), with lr ≈ 0.01. Simple, stable, and dependency-free.
   - (A hill-climbing/mutation variant is kept as a fallback if gradients prove noisy, but policy gradient is the default.)

3. Config naming: standardize on Section 4 strings — LEFT_CONTROLLER / RIGHT_CONTROLLER = "Human" | "AI_Model" | "Training". The Section 2 names ("Static_AI"/"Training_AI") are treated as descriptive labels only, not config values.

4. Generation length: a generation ends after 20 total ball resets (misses). At the boundary: record hit count and accuracy, append to history, reset counters, continue. Two "Training" paddles simultaneously is supported — each keeps its own brain, reward tally, and history.

5. Constants: 800x600 window, paddle 12x100 px moving 6 px/frame, ball 10x10 px starting at 5 px/frame and speeding up ~5% per paddle hit (capped), running at 60 FPS.

6. Ball reset: spawn at center (W/2, H/2), random vertical angle in roughly ±45°, horizontal direction serving toward the side that just conceded; base speed reset to 5 px/frame.

7. Accuracy metric: hits / (hits + misses) per generation, plotted as a line vs. generation number with matplotlib on exit. Saved next to the brain as <save_name>_history.png and also shown in a window.

8. Saving with 'S': auto-save the active Training brain to the configured save_name (e.g. 'defensive_bot.pkl') with no prompt, and print a console confirmation like "Saved brain to defensive_bot.pkl (gen 7, acc 0.82)". If both paddles are Training, save each to its own configured path.