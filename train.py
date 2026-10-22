"""Headless Q-Learning training against a random opponent."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from agent import QLearningAgent
from game import O, X, TicTacToe


@dataclass
class TrainingResult:
    episodes: int
    wins: int
    losses: int
    draws: int
    q_values: int
    final_epsilon: float


def _reward(result: int | str | None, player: int) -> float:
    if result == player:
        return 1.0
    if result == "draw":
        return 0.25
    return -1.0


def train_episode(agent: QLearningAgent, rng: random.Random, player: int) -> int | str:
    """Train one episode with the agent playing ``player``."""
    game = TicTacToe()

    while not game.done:
        if game.current_player == player:
            state = agent.encode_state(game.get_state(), player)
            action = agent.select_action(state, game.get_legal_actions())
            if not game.make_move(action):
                raise RuntimeError("Agent selected an illegal action.")

            if game.done:
                agent.update(state, action, _reward(game.get_result(), player), None, None, True)
                break

            opponent_action = rng.choice(game.get_legal_actions())
            game.make_move(opponent_action)
            if game.done:
                agent.update(state, action, _reward(game.get_result(), player), None, None, True)
            else:
                next_state = agent.encode_state(game.get_state(), player)
                agent.update(
                    state,
                    action,
                    0.0,
                    next_state,
                    game.get_legal_actions(),
                    False,
                )
        else:
            opponent_action = rng.choice(game.get_legal_actions())
            game.make_move(opponent_action)

    agent.decay_epsilon()
    return game.get_result()


def train(
    episodes: int = 50_000,
    seed: int | None = 7,
    agent: QLearningAgent | None = None,
) -> tuple[QLearningAgent, TrainingResult]:
    """Train against a random opponent, alternating the agent's side."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    rng = random.Random(seed)
    agent = agent or QLearningAgent(seed=seed)
    wins = losses = draws = 0

    for episode in range(episodes):
        player = X if episode % 2 == 0 else O
        result = train_episode(agent, rng, player)
        if result == player:
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1

    return agent, TrainingResult(
        episodes=episodes,
        wins=wins,
        losses=losses,
        draws=draws,
        q_values=len(agent.q_table),
        final_epsilon=agent.epsilon,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("q_table.json"))
    args = parser.parse_args()

    agent, result = train(args.episodes, args.seed)
    agent.save(args.output)
    print(
        f"Trained {result.episodes} episodes: "
        f"{result.wins} wins, {result.losses} losses, {result.draws} draws; "
        f"{result.q_values} Q-values saved to {args.output}."
    )


if __name__ == "__main__":
    main()
