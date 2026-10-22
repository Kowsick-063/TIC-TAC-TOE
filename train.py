"""Headless training and evaluation for the Tic-Tac-Toe Q-Learning agent."""

import argparse
import random
from functools import lru_cache

from agent import QLearningAgent
from game import O, X, TicTacToe

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def random_policy(game):
    return random.choice(game.get_legal_actions())


def _winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


@lru_cache(maxsize=None)
def _minimax(board, player):
    winner = _winner(board)
    if winner:
        return 1 if winner == player else -1
    if 0 not in board:
        return 0
    return max(
        -_minimax(board[:action] + (player,) + board[action + 1:], -player)
        for action in range(9)
        if board[action] == 0
    )


def optimal_policy(game):
    scored = []
    board = game.get_state()
    player = game.current_player
    for action in game.get_legal_actions():
        next_board = board[:action] + (player,) + board[action + 1:]
        scored.append((-_minimax(next_board, -player), action))
    best = max(score for score, _ in scored)
    return random.choice([action for score, action in scored if score == best])


def play_episode(agent, learners, opponent_policy=random_policy, training=True):
    """Play one game; learners is a list containing X, O, or both."""
    game = TicTacToe()
    pending = {player: None for player in learners}
    totals = {player: 0.0 for player in learners}

    while not game.done:
        player = game.current_player
        if player in learners:
            board = game.get_state()
            legal = game.get_legal_actions()
            action = agent.choose_action(board, legal, player, training)
            pending[player] = (board, action)
            game.make_move(action)
        else:
            game.make_move(opponent_policy(game))

        if training and not game.done:
            for learner in learners:
                if pending[learner] is not None and game.current_player == learner:
                    board, action = pending[learner]
                    agent.update(
                        board, action, 0.0, game.get_state(),
                        game.get_legal_actions(), False, learner
                    )
                    pending[learner] = None

    result = game.get_result()
    for learner in learners:
        reward = agent.get_reward(result, learner)
        totals[learner] += reward
        if training and pending[learner] is not None:
            board, action = pending[learner]
            agent.update(board, action, reward, game.get_state(), [], True, learner)
    return result, totals


def evaluate(agent, opponent_policy=random_policy, games=1000):
    if games < 1:
        raise ValueError("games must be positive.")
    wins = losses = draws = 0
    reward_sum = 0.0
    for index in range(games):
        player = X if index % 2 == 0 else O
        result, totals = play_episode(
            agent, [player], opponent_policy=opponent_policy, training=False
        )
        reward_sum += totals[player]
        if result == player:
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": 100 * wins / games,
        "loss_rate": 100 * losses / games,
        "draw_rate": 100 * draws / games,
        "avg_reward": reward_sum / games,
    }


def train(agent, episodes=50000, opponent="random", checkpoint_every=1000,
          eval_games=200):
    """Train against random, minimax, or self-play and return checkpoints."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    policies = {"random": random_policy, "optimal": optimal_policy}
    log = []
    for episode in range(1, episodes + 1):
        if opponent == "self":
            learners, policy = [X, O], random_policy
        elif opponent in policies:
            player = X if episode % 2 else O
            learners, policy = [player], policies[opponent]
        else:
            raise ValueError("opponent must be random, optimal, or self.")
        play_episode(agent, learners, policy, training=True)
        agent.decay_epsilon()
        if episode % checkpoint_every == 0:
            stats = evaluate(agent, random_policy, eval_games)
            stats.update(episode=episode, epsilon=agent.epsilon, states=len(agent))
            log.append(stats)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--opponent", choices=("random", "optimal", "self"),
                        default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="q_table.pkl")
    args = parser.parse_args()
    random.seed(args.seed)
    agent = QLearningAgent(seed=args.seed)
    log = train(agent, args.episodes, args.opponent,
                max(1, args.episodes // 25), 200)
    agent.save(args.out)
    result = evaluate(agent, random_policy, 1000)
    print(
        f"Trained {args.episodes} episodes; "
        f"random-baseline win/loss/draw = "
        f"{result['wins']}/{result['losses']}/{result['draws']}; "
        f"{len(agent)} states saved to {args.out}."
    )
    return log


if __name__ == "__main__":
    main()
