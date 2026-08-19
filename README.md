# self-play-tictactoe

A Tic-Tac-Toe AI that learns to play entirely through self-play reinforcement learning &mdash; no hardcoded strategy, no training data. Two copies of the same Q-learning agent play against each other thousands of times, and each game's outcome (win, loss, or draw) is used to update their strategy.

Since Tic-Tac-Toe is a solved game &mdash; perfect play from both sides always ends in a draw &mdash; the real signal of learning here isn't win rate, it's watching the **draw rate climb** as the agent gets better at avoiding mistakes.

## How it works

1. **Q-learning**: each agent keeps a table mapping `(board state, move)` pairs to an expected value. Moves with a higher expected value are more likely to be chosen.
2. **Self-play**: two agents play full games against each other. Early on, moves are mostly random (high exploration); over time, exploration is gradually reduced so the agents rely more on what they've learned.
3. **Reward propagation**: at the end of each game, the result (win / loss / draw) is propagated backward through every move the agent made that game, so earlier moves that contributed to a win get credited too, not just the final one.
4. **Repeat thousands of times**: over 50,000 games, the agents converge toward near-optimal play, and draws become the dominant outcome.

## Project structure

- `game.py` &mdash; core Tic-Tac-Toe rules and board logic, no AI involved
- `agent.py` &mdash; the Q-learning agent: how it chooses moves and learns from results
- `train.py` &mdash; runs the self-play training loop, tracks win/draw rates, saves the trained agents and a progress chart
- `play.py` &mdash; lets you play against the trained agent yourself in the terminal

## Setup

Requires Python 3 and one dependency (`matplotlib`, used only for the training progress chart).

```bash
pip3 install matplotlib
```

If that fails with a "externally managed environment" error:

```bash
pip3 install matplotlib --break-system-packages
```

## Usage

**Train the agent from scratch:**

```bash
python3 train.py
```

This runs 50,000 self-play games, printing win/draw counts every 500 games. It takes under a minute on most machines. When finished, it saves:
- `agent_x.pkl` / `agent_o.pkl` &mdash; the trained agents' learned Q-tables
- `training_progress.png` &mdash; a chart of win rate and draw rate over the course of training

**Play against the trained agent:**

```bash
python3 play.py
```

You play as O; the trained agent plays as X and moves first. Enter the position number (0&ndash;8) for your move each turn:

```
0 1 2
3 4 5
6 7 8
```

## What to expect

Early in training, the agents win and lose against each other fairly evenly, since moves are close to random. As training progresses, the draw rate rises steadily and eventually dominates &mdash; by the end of training, the majority of games between the two agents end in a draw, which is the mathematically correct outcome for two players who never make a mistake.

## Notes

- The agents use a simple table-based Q-learning approach, not a neural network &mdash; this keeps the project easy to understand end-to-end while still demonstrating the core reinforcement learning loop (act, observe outcome, update strategy, repeat).
- Retraining (`python3 train.py`) overwrites the saved agent files, so you can experiment with the hyperparameters in `train.py` (number of games, learning rate, exploration rate) and re-run freely.
