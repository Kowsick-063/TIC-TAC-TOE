"""Evaluate a saved Q-Learning agent against a random opponent."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from agent import QLearningAgent
from game import O, X, TicTacToe


@dataclass
class EvaluationResult:
    games: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games


def evaluate(
    agent: QLearningAgent,
    games: int = 1_000,
    seed: int | None = 11,
) -> EvaluationResult:
    """Run independent greedy games, alternating X and O."""
    if games < 1:
        raise ValueError("games must be positive.")
    rng = random.Random(seed)
    wins = losses = draws = 0

    for game_number in range(games):
        player = X if game_number % 2 == 0 else O
        game = TicTacToe()
        while not game.done:
            if game.current_player == player:
                state = agent.encode_state(game.get_state(), player)
                action = agent.select_action(
                    state, game.get_legal_actions(), training=False
                )
            else:
                action = rng.choice(game.get_legal_actions())
            if not game.make_move(action):
                raise RuntimeError("Evaluation selected an illegal action.")

        result = game.get_result()
        if result == player:
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1

    return EvaluationResult(games, wins, losses, draws)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("q_table.json"))
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    agent = QLearningAgent.load(args.model, seed=args.seed)
    result = evaluate(agent, args.games, args.seed)
    print(
        f"{result.games} games: {result.wins} wins ({result.win_rate:.1%}), "
        f"{result.losses} losses ({result.loss_rate:.1%}), "
        f"{result.draws} draws ({result.draw_rate:.1%})."
    )


if __name__ == "__main__":
    main()
