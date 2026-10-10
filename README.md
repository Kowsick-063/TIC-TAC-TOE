
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
├── test_game.py
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
cd TicTacToe-RL
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

## Run the Game

Start the graphical application:

```bash
python main.py
```

## Run Tests

Run the game-logic tests:

```bash
python test_game.py
```

## How to Play

1. Launch the application.
2. Select Human vs AI or Human vs Human.
3. Click an empty cell to place your mark.
4. Complete a row, column, or diagonal to win.
5. Click New Game to start another match.

In Human vs AI mode, the human plays X and the temporary AI plays O.

## Current Implementation Status

- [x] Game board and move validation
- [x] Win and draw detection
- [x] Pygame graphical interface
- [x] Human vs Human mode
- [x] Temporary random-move AI
- [x] Session-based statistics and leaderboard
- [ ] Q-Learning agent integration
- [ ] AI training pipeline
- [ ] Evaluation of the trained agent
- [ ] Persistent leaderboard and saved model

## Reinforcement Learning Plan

The intended Q-Learning agent will learn a policy through repeated games.

The planned components are:

- **State:** The 3 × 3 board configuration
- **Actions:** Legal empty board positions
- **Rewards:** Positive for a win, negative for a loss, and a suitable reward for a draw
- **Learning:** Q-value updates using the Q-Learning algorithm
- **Evaluation:** Win, loss, and draw rates against a defined opponent

The current random-move opponent is for interface testing only. Its performance should not be presented as trained Reinforcement Learning performance.

## Team

Developed as a student mini-project on Reinforcement Learning.

## License

This project is intended for educational purposes.