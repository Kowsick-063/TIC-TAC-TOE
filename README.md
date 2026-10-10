
# Tic-Tac-Toe RL

A Python-based Tic-Tac-Toe game developed as a mini-project to explore Reinforcement Learning.

The project includes a graphical interface built with Pygame, two-player gameplay, a temporary random-move AI opponent, and session-based game statistics.

## Features

- Interactive 3 × 3 Tic-Tac-Toe board
- Minimalistic light-themed graphical interface
- Human vs AI mode
- Human vs Human mode
- Automatic win and draw detection
- Session leaderboard and statistics
- Restart functionality
- Modular game logic for future Reinforcement Learning integration

## Technology Stack

- Python
- Pygame
- Reinforcement Learning (planned Q-Learning integration)

## Project Structure

```text
TicTacToe-RL/
├── main.py
├── game.py
├── agent.py
├── train.py
├── test_game.py
├── test_agent.py
├── test_train.py
├── q_table.pkl
├── results.csv
├── training_log.csv
├── learning_curve.png
├── epsilon_curve.png
├── outcomes.png
├── REPORT_MEMBER2.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or later
- pip
- Visual Studio Code or another Python-compatible IDE

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TIC-TAC-TOE
```

Alternatively, open your existing project folder in VS Code.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Train the AI (optional - q_table.pkl is already included)

```bash
python train.py
```

Options: `--episodes 100000`, `--opponent random|self|optimal|noisy|mixed`, `--seed 7`.
Training plays 60,000 games (about 1-2 minutes), evaluates before/after, and writes
`q_table.pkl`, `results.csv`, `training_log.csv` and three graphs.

## Run the Game

Start the graphical application:

```bash
python main.py
```

## Run Tests

```bash
python test_game.py     # game logic
python test_agent.py    # Q-Learning agent (~15 s, includes a short training run)
python test_train.py    # training / evaluation / saved model
```

## How to Play

1. Launch the application.
2. Select Human vs AI or Human vs Human.
3. Click an empty cell to place your mark.
4. Complete a row, column, or diagonal to win.
5. Click New Game to start another match.

In Human vs AI mode, the human plays X and the trained Q-Learning agent plays O.

## Current Implementation Status

- [x] Game board and move validation
- [x] Win and draw detection
- [x] Pygame graphical interface
- [x] Human vs Human mode
- [x] Temporary random-move AI (replaced by the trained agent)
- [x] Session-based statistics and leaderboard
- [x] Q-Learning agent (`agent.py`, Member 2)
- [x] AI training pipeline (`train.py`)
- [x] Evaluation of the trained agent (`results.csv`, graphs)
- [x] Saved model (`q_table.pkl`)
- [ ] Persistent leaderboard

## Reinforcement Learning Plan

The intended Q-Learning agent will learn a policy through repeated games.

The planned components are:

- **State:** The 3 × 3 board configuration
- **Actions:** Legal empty board positions
- **Rewards:** Positive for a win, negative for a loss, and a suitable reward for a draw
- **Learning:** Q-value updates using the Q-Learning algorithm
- **Evaluation:** Win, loss, and draw rates against a defined opponent


## Team

Developed as a student mini-project on Reinforcement Learning.

## License

This project is intended for educational purposes.

## Q-Learning Agent (Member 2) - `agent.py`

| Item | Design |
|---|---|
| State | Board tuple of 9 values from the agent's own perspective (own marks = +1, opponent = -1), so one Q-table serves both X and O |
| Action | Index 0-8 of an empty cell (only legal actions are ever chosen) |
| Rewards | Win +1, Draw +0.2, Loss -1, Ongoing 0 |
| Policy | Epsilon-greedy, epsilon decays per episode down to `epsilon_min` |
| Update | `Q(s,a) <- Q(s,a) + alpha [ r + gamma max Q(s',a') - Q(s,a) ]`, future term = 0 on terminal states |
| Defaults | alpha 0.5, gamma 0.9, epsilon 1.0 -> 0.05, decay 0.9995 |

### Interface (for Member 3's `train.py` and `main.py`)

```python
from agent import QLearningAgent

agent = QLearningAgent()

# choose a move (training=True -> epsilon-greedy, False -> greedy)
action = agent.choose_action(game.get_state(), game.get_legal_actions(),
                             game.current_player, training=True)

# learn from a transition
agent.update(board, action, reward, next_board,
             next_legal_actions, done, player)

reward = agent.get_reward(game.get_result(), player)   # +1 / 0.2 / -1 / 0
agent.decay_epsilon()                                  # once per episode
agent.save("q_table.pkl");  agent = QLearningAgent.load("q_table.pkl")
```

**Important for the training loop:** after the agent moves, wait for the
opponent's reply, then call `update` with the board where the agent moves
*next* (the opponent is part of the environment). If the game ends on either
side's move, call `update` with `done=True` and the final reward. See
`play_episode()` in `test_agent.py` for a complete working example.

### Plugging the trained AI into `main.py`

Replace the body of `ai_move()` with:

```python
agent = QLearningAgent.load("q_table.pkl")   # once, at start-up

def ai_move():
    if mode == "ai" and not game.done and game.current_player == O:
        action = agent.choose_action(game.get_state(),
                                     game.get_legal_actions(), O,
                                     training=False)
        game.make_move(action)
```

## Training and Evaluation (Member 3) - `train.py`

**Training:** 60,000 episodes of a mixed schedule - 30% self-play (one shared Q-table
plays both X and O), 20% vs a random player, 20% vs a perfect minimax player,
30% vs a "noisy" perfect player (25% random moves). Epsilon decays from 1.0 to 0.05.

**Evaluation:** greedy play (no learning), 1,000 games per opponent, agent alternating X and O.

| Opponent | Stage | Wins | Losses | Draws | Win % | Loss % | Avg reward |
|---|---|---|---|---|---|---|---|
| Random | Before | 432 | 436 | 132 | 43.2 | 43.6 | 0.022 |
| Random | After | 920 | 0 | 80 | 92.0 | 0.0 | 0.936 |
| Optimal (minimax) | Before | 0 | 877 | 123 | 0.0 | 87.7 | -0.852 |
| Optimal (minimax) | After | 0 | 0 | 1000 | 0.0 | 0.0 | 0.200 |

Against a perfect player the best possible result is a draw; the trained agent reaches it
in every game. An exhaustive check of every possible human move sequence also found no
line in which the agent loses (579 game lines as O, 108 as X). Exact numbers vary slightly
with the random seed.
