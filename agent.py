"""Perspective-consistent tabular Q-Learning for Tic-Tac-Toe.

The Q-table maps a board state, represented from the learning player's
perspective, to nine action values. Own marks are ``1``, opponent marks are
``-1``, and empty cells are ``0``. Actions are board positions 0 through 8.
Models are stored as JSON, even when a legacy ``.pkl`` filename is supplied;
JSON avoids unsafe deserialization.
"""

import json
import math
import random
from pathlib import Path

REWARD_WIN = 1.0
REWARD_DRAW = 0.2
REWARD_LOSS = -1.0
REWARD_STEP = 0.0
DRAW = "draw"


class QLearningAgent:
    MODEL_VERSION = 1

    def __init__(
        self,
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 <= epsilon <= 1 or not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon and epsilon_min must be in [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self._random = random.Random(seed)

    @staticmethod
    def encode_state(board, player):
        """Return ``board`` from ``player``'s perspective."""
        if player not in (-1, 1):
            raise ValueError("player must be X (1) or O (-1).")
        board = tuple(board)
        if len(board) != 9 or any(cell not in (-1, 0, 1) for cell in board):
            raise ValueError("board must contain nine values in {-1, 0, 1}.")
        return tuple(cell * player for cell in board)

    @staticmethod
    def _validate_state(state):
        state = tuple(state)
        if len(state) != 9 or any(cell not in (-1, 0, 1) for cell in state):
            raise ValueError("state must contain nine values in {-1, 0, 1}.")
        return state

    @staticmethod
    def get_legal_actions(board):
        return [index for index, cell in enumerate(board) if cell == 0]

    def _q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        return self.q_table[state]

    def get_q(self, board, action, player):
        if action not in range(9):
            raise ValueError("action must be between 0 and 8.")
        return self._q_values(self.encode_state(board, player))[action]

    def choose_action(self, board, legal_actions=None, player=1, training=True):
        """Choose a legal action using epsilon-greedy selection."""
        board = tuple(board)
        legal_actions = (
            self.get_legal_actions(board)
            if legal_actions is None
            else list(legal_actions)
        )
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if any(action not in range(9) for action in legal_actions):
            raise ValueError("legal_actions contains an invalid action.")
        if training and self._random.random() < self.epsilon:
            return self._random.choice(legal_actions)
        values = self._q_values(self.encode_state(board, player))
        best_value = max(values[action] for action in legal_actions)
        best = [action for action in legal_actions if values[action] == best_value]
        return self._random.choice(best)

    def select_action(self, state, legal_actions, training=True):
        """Select an action when ``state`` is already perspective-encoded."""
        state = self._validate_state(state)
        return self.choose_action(state, legal_actions, 1, training)

    def best_action(self, board, legal_actions, player):
        return self.choose_action(board, legal_actions, player, False)

    @staticmethod
    def get_reward(result, player):
        if result is None:
            return REWARD_STEP
        if result == DRAW:
            return REWARD_DRAW
        return REWARD_WIN if result == player else REWARD_LOSS

    def update(
        self, board, action, reward, next_board, next_legal_actions, done, player=1
    ):
        """Apply Q(s,a) <- Q(s,a) + alpha(target - Q(s,a))."""
        state = self.encode_state(board, player)
        values = self._q_values(state)
        if action not in range(9):
            raise ValueError("action must be between 0 and 8.")
        if done:
            future = 0.0
        else:
            if next_board is None or not next_legal_actions:
                raise ValueError("Non-terminal updates need legal next actions.")
            next_state = self.encode_state(next_board, player)
            next_values = self._q_values(next_state)
            future = max(next_values[a] for a in next_legal_actions)
        target = reward + self.gamma * future
        values[action] += self.alpha * (target - values[action])
        return values[action]

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def __len__(self):
        return len(self.q_table)

    def save(self, path="q_table.pkl"):
        """Save model data as validated JSON rather than executable pickle data."""
        payload = {
            "version": self.MODEL_VERSION,
            "parameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in self.q_table.items()
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path="q_table.pkl"):
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read model {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != cls.MODEL_VERSION:
            raise ValueError("Unsupported Q-table model.")
        parameters = payload.get("parameters", {})
        agent = cls(**{
            key: parameters[key]
            for key in ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay")
            if key in parameters
        })
        entries = payload.get("q_table")
        if not isinstance(entries, list):
            raise ValueError("Model q_table must be a list.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid Q-table entry.")
            state = cls._validate_state(entry.get("state", []))
            values = entry.get("values")
            if not isinstance(values, list) or len(values) != 9:
                raise ValueError("Each Q-table entry needs nine values.")
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in values):
                raise ValueError("Q-table values must be finite numbers.")
            agent.q_table[state] = [float(value) for value in values]
        return agent
