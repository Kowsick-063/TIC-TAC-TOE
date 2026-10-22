"""Tabular Q-Learning agent for Tic-Tac-Toe.

States passed to this class are always from the learning player's
perspective: the learning player's marks are ``1``, the opponent's marks
are ``-1``, and empty cells are ``0``. Actions are board positions 0-8.
"""

import json
import math
import random
from pathlib import Path
from typing import Iterable


State = tuple[int, ...]
QTable = dict[tuple[State, int], float]


class QLearningAgent:
    """A legal-action epsilon-greedy tabular Q-Learning agent."""

    MODEL_VERSION = 1

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in the interval [0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in the interval [0, 1].")
        if not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon_min must be in the interval [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in the interval (0, 1].")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._random = random.Random(seed)
        self.q_table: QTable = {}

    @staticmethod
    def encode_state(board: Iterable[int], player: int) -> State:
        """Encode an absolute board from ``player``'s point of view."""
        if player not in (-1, 1):
            raise ValueError("player must be 1 (X) or -1 (O).")
        state = tuple(int(cell) * player for cell in board)
        if len(state) != 9 or any(cell not in (-1, 0, 1) for cell in state):
            raise ValueError("board must contain exactly nine values in {-1, 0, 1}.")
        return state

    @staticmethod
    def _validate_state(state: Iterable[int]) -> State:
        state = tuple(state)
        if len(state) != 9 or any(cell not in (-1, 0, 1) for cell in state):
            raise ValueError("state must contain exactly nine values in {-1, 0, 1}.")
        return state

    @staticmethod
    def _validate_actions(legal_actions: Iterable[int]) -> list[int]:
        actions = list(legal_actions)
        if not actions or any(action not in range(9) for action in actions):
            raise ValueError("legal_actions must contain at least one board position from 0 to 8.")
        if len(set(actions)) != len(actions):
            raise ValueError("legal_actions must not contain duplicates.")
        return actions

    def _value(self, state: State, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def select_action(
        self,
        state: Iterable[int],
        legal_actions: Iterable[int],
        training: bool = True,
    ) -> int:
        """Select a legal action with epsilon-greedy exploration."""
        state = self._validate_state(state)
        actions = self._validate_actions(legal_actions)

        if training and self._random.random() < self.epsilon:
            return self._random.choice(actions)

        best_value = max(self._value(state, action) for action in actions)
        best_actions = [
            action for action in actions
            if math.isclose(self._value(state, action), best_value)
        ]
        return self._random.choice(best_actions)

    def update(
        self,
        state: Iterable[int],
        action: int,
        reward: float,
        next_state: Iterable[int] | None,
        legal_next_actions: Iterable[int] | None,
        done: bool,
    ) -> float:
        """Apply one Q-Learning update and return the new Q-value.

        Terminal transitions never bootstrap. A non-terminal transition must
        provide at least one legal next action so that the maximum is valid.
        """
        state = self._validate_state(state)
        if action not in range(9):
            raise ValueError("action must be a board position from 0 to 8.")
        if done:
            target = float(reward)
        else:
            if next_state is None or legal_next_actions is None:
                raise ValueError("non-terminal updates require next state and legal actions.")
            next_state = self._validate_state(next_state)
            actions = self._validate_actions(legal_next_actions)
            target = float(reward) + self.gamma * max(
                self._value(next_state, next_action) for next_action in actions
            )

        old_value = self._value(state, action)
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[(state, action)] = new_value
        return new_value

    def decay_epsilon(self) -> float:
        """Decay exploration after an episode and return the new epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> None:
        """Save a validated, portable JSON model."""
        path = Path(path)
        records = [
            {"state": list(state), "action": action, "value": value}
            for (state, action), value in self.q_table.items()
        ]
        payload = {
            "version": self.MODEL_VERSION,
            "parameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_values": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "QLearningAgent":
        """Load and validate a JSON model created by :meth:`save`."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model file {path}: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("version") != cls.MODEL_VERSION:
            raise ValueError("Unsupported or invalid Q-table model version.")
        parameters = payload.get("parameters")
        records = payload.get("q_values")
        if not isinstance(parameters, dict) or not isinstance(records, list):
            raise ValueError("Model must contain parameters and q_values.")

        agent = cls(seed=seed, **{
            name: parameters[name]
            for name in (
                "alpha",
                "gamma",
                "epsilon",
                "epsilon_min",
                "epsilon_decay",
            )
            if name in parameters
        })
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each Q-table entry must be an object.")
            state = cls._validate_state(record.get("state", []))
            action = record.get("action")
            value = record.get("value")
            if not isinstance(action, int) or action not in range(9):
                raise ValueError("Each Q-table action must be an integer from 0 to 8.")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Each Q-table value must be finite.")
            agent.q_table[(state, action)] = float(value)
        return agent
