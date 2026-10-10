"""
train.py  -  Member 3: Training, evaluation and integration
============================================================

Usage
-----
    python train.py                       # default: 60,000 mixed episodes
    python train.py --episodes 100000
    python train.py --opponent self       # random | self | optimal | noisy | mixed
    python train.py --seed 7

Outputs (in this folder)
------------------------
    q_table.pkl        trained Q-table  (loaded by main.py)
    results.csv        before / after evaluation table
    training_log.csv   evaluation checkpoints during training
    learning_curve.png win/loss/draw rate while training   (needs matplotlib)
    epsilon_curve.png  exploration rate over training       (needs matplotlib)
    outcomes.png       before vs after bar chart            (needs matplotlib)
"""

import argparse
import csv
import random
from functools import lru_cache

from game import TicTacToe, X, O
from agent import QLearningAgent

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


# ----------------------------------------------------------------------
# Opponents
# ----------------------------------------------------------------------
def random_policy(game):
    """Plays any legal move."""
    return random.choice(game.get_legal_actions())


def _winner(board):
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


@lru_cache(maxsize=None)
def _minimax(board, player):
    """Value of `board` for the player to move (+1 win, 0 draw, -1 loss)."""
    w = _winner(board)
    if w != 0:
        return 1 if w == player else -1
    if 0 not in board:
        return 0
    best = -2
    for i in range(9):
        if board[i] == 0:
            nb = board[:i] + (player,) + board[i + 1:]
            best = max(best, -_minimax(nb, -player))
    return best


def optimal_policy(game):
    """Perfect player (minimax). Picks randomly among equally good moves."""
    board, player = game.get_state(), game.current_player
    scored = []
    for i in game.get_legal_actions():
        nb = board[:i] + (player,) + board[i + 1:]
        scored.append((-_minimax(nb, -player), i))
    best = max(s for s, _ in scored)
    return random.choice([i for s, i in scored if s == best])


def noisy_optimal_policy(game, noise=0.25):
    """Perfect play, but with probability `noise` a random move - this
    exposes the agent to many different positions during training."""
    if random.random() < noise:
        return random_policy(game)
    return optimal_policy(game)


# ----------------------------------------------------------------------
# One game
# ----------------------------------------------------------------------
def play_episode(agent, learners, opponent_policy=None, training=True):
    """
    Play one game.
      learners        : marks controlled by `agent` (e.g. [X], [O] or [X, O]
                        for self-play with a shared Q-table).
      opponent_policy : function(game) -> action for any other mark.
    Returns (result, {mark: total_reward}).
    """
    game = TicTacToe()
    pending = {p: None for p in learners}   # (board, action) awaiting next state
    totals = {p: 0.0 for p in learners}

    while not game.done:
        player = game.current_player
        if player in learners:
            board, legal = game.get_state(), game.get_legal_actions()
            if training and pending[player] is not None:
                b, a = pending[player]
                agent.update(b, a, 0.0, board, legal, False, player)
            action = agent.choose_action(board, legal, player, training)
            pending[player] = (board, action)
            game.make_move(action)
        else:
            game.make_move(opponent_policy(game))

    result = game.get_result()
    for p in learners:
        reward = agent.get_reward(result, p)
        totals[p] += reward
        if training and pending[p] is not None:
            b, a = pending[p]
            agent.update(b, a, reward, game.get_state(), [], True, p)
    return result, totals


# ----------------------------------------------------------------------
# Evaluation (greedy, no learning)
# ----------------------------------------------------------------------
def evaluate(agent, opponent_policy, games=1000):
    """Agent plays X in half the games and O in the other half."""
    wins = losses = draws = 0
    reward_sum = 0.0
    for i in range(games):
        me = X if i % 2 == 0 else O
        result, totals = play_episode(agent, [me], opponent_policy,
                                      training=False)
        reward_sum += totals[me]
        if result == "draw":
            draws += 1
        elif result == me:
            wins += 1
        else:
            losses += 1
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": 100 * wins / games,
        "loss_rate": 100 * losses / games,
        "draw_rate": 100 * draws / games,
        "avg_reward": reward_sum / games,
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def pick_mode(opponent):
    """Choose how one training episode is played."""
    if opponent == "mixed":
        r = random.random()
        opponent = ("self" if r < 0.3 else "random" if r < 0.5
                    else "optimal" if r < 0.7 else "noisy")
    if opponent == "self":
        return [X, O], None                      # both sides learn
    me = random.choice([X, O])                   # agent vs a fixed opponent
    policy = {"random": random_policy, "optimal": optimal_policy,
              "noisy": noisy_optimal_policy}[opponent]
    return [me], policy


def train(agent, episodes, opponent, checkpoint_every, eval_games):
    log = []
    wins = losses = draws = 0
    for ep in range(1, episodes + 1):
        learners, policy = pick_mode(opponent)
        result, _ = play_episode(agent, learners, policy, training=True)
        agent.decay_epsilon()

        if ep % checkpoint_every == 0:
            stats = evaluate(agent, random_policy, eval_games)
            stats.update(episode=ep, epsilon=agent.epsilon,
                         states=len(agent))
            log.append(stats)
            print(f"  episode {ep:>7}  eps={agent.epsilon:.3f}  "
                  f"vs random: win {stats['win_rate']:5.1f}%  "
                  f"loss {stats['loss_rate']:4.1f}%  "
                  f"draw {stats['draw_rate']:4.1f}%  "
                  f"states={len(agent)}")
    return log


# ----------------------------------------------------------------------
# Saving results / graphs
# ----------------------------------------------------------------------
def save_results(rows, path="results.csv"):
    fields = ["opponent", "stage", "games", "wins", "losses", "draws",
              "win_rate", "loss_rate", "draw_rate", "avg_reward"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 3) if isinstance(r[k], float) else r[k])
                        for k in fields})


def save_log(log, path="training_log.csv"):
    fields = ["episode", "epsilon", "states", "win_rate", "loss_rate",
              "draw_rate", "avg_reward"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in log:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in fields})


def make_graphs(log, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping graphs "
              "(pip install matplotlib).")
        return

    eps = [r["episode"] for r in log]

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, [r["win_rate"] for r in log], label="Win %", color="#2e9d69")
    plt.plot(eps, [r["draw_rate"] for r in log], label="Draw %", color="#3970e8")
    plt.plot(eps, [r["loss_rate"] for r in log], label="Loss %", color="#de5b71")
    plt.xlabel("Training episodes")
    plt.ylabel("Percent of 500 evaluation games vs random")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, [r["epsilon"] for r in log], color="#7d8796")
    plt.xlabel("Training episodes")
    plt.ylabel("Epsilon (exploration rate)")
    plt.title("Exploration decay")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("epsilon_curve.png", dpi=150)
    plt.close()

    labels = [f"{r['opponent']}\n{r['stage']}" for r in rows]
    x = range(len(rows))
    plt.figure(figsize=(9, 4.8))
    plt.bar([i - 0.27 for i in x], [r["win_rate"] for r in rows], 0.27,
            label="Win %", color="#2e9d69")
    plt.bar(list(x), [r["draw_rate"] for r in rows], 0.27,
            label="Draw %", color="#3970e8")
    plt.bar([i + 0.27 for i in x], [r["loss_rate"] for r in rows], 0.27,
            label="Loss %", color="#de5b71")
    plt.xticks(list(x), labels)
    plt.ylabel("Percent of games")
    plt.title("Before vs after training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outcomes.png", dpi=150)
    plt.close()
    print("Saved learning_curve.png, epsilon_curve.png, outcomes.png")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the Tic-Tac-Toe Q-agent")
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--opponent", choices=["random", "self", "optimal", "noisy", "mixed"],
                    default="mixed",
                    help="random | self (self-play) | optimal (minimax) | "
                         "noisy (perfect play with 25%% random moves) | "
                         "mixed (30%% self-play, 20%% random, 20%% optimal, "
                         "30%% noisy)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--epsilon-min", type=float, default=0.05)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="q_table.pkl")
    args = ap.parse_args()

    random.seed(args.seed)

    # Decay so epsilon reaches its floor after ~60% of training.
    decay = args.epsilon_min ** (1.0 / (0.6 * args.episodes))
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=1.0,
                           epsilon_min=args.epsilon_min, epsilon_decay=decay)

    rows = []
    print("Evaluating BEFORE training...")
    for name, policy in (("random", random_policy), ("optimal", optimal_policy)):
        r = evaluate(agent, policy, args.eval_games)
        r.update(opponent=name, stage="before")
        rows.append(r)

    print(f"\nTraining for {args.episodes} episodes "
          f"(opponent mode: {args.opponent})...")
    log = train(agent, args.episodes,
                args.opponent, max(1, args.episodes // 30), 500)

    print("\nEvaluating AFTER training...")
    for name, policy in (("random", random_policy), ("optimal", optimal_policy)):
        r = evaluate(agent, policy, args.eval_games)
        r.update(opponent=name, stage="after")
        rows.append(r)

    agent.save(args.out)
    save_results(rows)
    save_log(log)
    make_graphs(log, rows)

    print(f"\n{'Opponent':<9}{'Stage':<8}{'Games':>6}{'Wins':>6}{'Losses':>8}"
          f"{'Draws':>7}{'Win%':>7}{'Loss%':>7}{'AvgRew':>8}")
    for r in rows:
        print(f"{r['opponent']:<9}{r['stage']:<8}{r['games']:>6}{r['wins']:>6}"
              f"{r['losses']:>8}{r['draws']:>7}{r['win_rate']:>7.1f}"
              f"{r['loss_rate']:>7.1f}{r['avg_reward']:>8.3f}")
    print(f"\nSaved Q-table ({len(agent)} states) to {args.out}, "
          f"results.csv and training_log.csv")


if __name__ == "__main__":
    main()
