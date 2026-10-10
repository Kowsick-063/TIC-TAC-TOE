"""
agent.py  -  Member 2: Q-Learning agent for Tic-Tac-Toe
=========================================================

Public interface (agreed with game.py / train.py / main.py)
-----------------------------------------------------------
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=1.0,
                           epsilon_min=0.05, epsilon_decay=0.9995)

    agent.get_legal_actions(board)                                    -> [int]
    agent.choose_action(board, legal_actions, player, training=True)  -> int (0..8)
    agent.update(board, action, reward, next_board,
                 next_legal_actions, done, player)                    -> None
    agent.get_reward(result, player)                                  -> float
    agent.decay_epsilon()                                             -> None
    agent.save(path) / QLearningAgent.load(path)

`board` is exactly what game.get_state() returns: a tuple of 9 values
(0 = empty, 1 = X, -1 = O). `player` is the mark the agent is playing
(game.X or game.O). `legal_actions` is game.get_legal_actions().

State representation
--------------------
The board is stored from the *agent's own perspective*: every cell is
multiplied by `player`, so the agent's marks are always +1 and the
opponent's marks are always -1. Because the agent only chooses a move on
its own turn, "whose turn it is" is implicit in the state. A side effect
is that one Q-table works for X and for O.

Q-table
-------
dict:  state (tuple of 9) -> list of 9 floats (one Q-value per cell).
Unseen states are created lazily with all Q-values = 0.
Illegal cells are never selected or used in the max, so their values
are irrelevant.

Q-Learning update
-----------------
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
For a terminal next state the bracket's future term is 0:
    Q(s,a) <- Q(s,a) + alpha * [ r - Q(s,a) ]
"""

import pickle
import random

# Reward design (from the agent's perspective)
REWARD_WIN = 1.0
REWARD_DRAW = 0.2
REWARD_LOSS = -1.0
REWARD_STEP = 0.0

DRAW = "draw"  # same value returned by TicTacToe.get_result()


class QLearningAgent:
    def __init__(
        self,
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
    ):
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # current exploration rate
        self.epsilon_min = epsilon_min      # floor for epsilon
        self.epsilon_decay = epsilon_decay  # multiplicative decay per episode
        self.q_table = {}

    # ------------------------------------------------------------------
    # State / Q-table helpers
    # ------------------------------------------------------------------
    @staticmethod
    def encode_state(board, player):
        """Board from the agent's point of view (own marks = +1)."""
        return tuple(cell * player for cell in board)

    @staticmethod
    def get_legal_actions(board):
        """Legal actions in a state: the indices (0-8) of the empty cells."""
        return [i for i, cell in enumerate(board) if cell == 0]

    def _q_values(self, state):
        """Return the 9 Q-values of a state, creating them if unseen."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        return self.q_table[state]

    def get_q(self, board, action, player):
        """Q-value of taking `action` on `board` for `player`."""
        return self._q_values(self.encode_state(board, player))[action]

    @staticmethod
    def _max_q(q_values, legal_actions):
        """Largest Q-value among legal actions only (0 if none: terminal)."""
        if not legal_actions:
            return 0.0
        return max(q_values[a] for a in legal_actions)

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------
    def choose_action(self, board, legal_actions=None, player=1, training=True):
        """
        training=True : epsilon-greedy (explore with probability epsilon).
        training=False: purely greedy, for evaluation and human-vs-AI play.
        Only legal actions are ever returned.
        """
        if legal_actions is None:
            legal_actions = self.get_legal_actions(board)
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)  # explore

        return self.best_action(board, legal_actions, player)  # exploit

    def best_action(self, board, legal_actions, player):
        """Legal action with the highest Q-value (ties broken randomly)."""
        q_values = self._q_values(self.encode_state(board, player))
        best_value = max(q_values[a] for a in legal_actions)
        best = [a for a in legal_actions if q_values[a] == best_value]
        return random.choice(best)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    @staticmethod
    def get_reward(result, player):
        """
        Reward for `player`, given game.get_result():
            player's mark -> win, 'draw' -> draw,
            opponent's mark -> loss, None -> game continues.
        """
        if result is None:
            return REWARD_STEP
        if result == DRAW:
            return REWARD_DRAW
        if result == player:
            return REWARD_WIN
        return REWARD_LOSS

    # ------------------------------------------------------------------
    # Q-Learning update
    # ------------------------------------------------------------------
    def update(self, board, action, reward, next_board,
               next_legal_actions, done, player):
        """
        Apply one Q-Learning update for the transition
            (board, action) -> reward, next_board

        `next_board` is the next position where the AGENT must move
        (i.e. after the opponent has replied), or the final board if the
        game ended. When done=True the future-value term is 0 (terminal
        state handling).
        """
        state = self.encode_state(board, player)
        q_values = self._q_values(state)

        if done:
            future = 0.0
        else:
            next_state = self.encode_state(next_board, player)
            future = self._max_q(self._q_values(next_state),
                                 next_legal_actions)

        target = reward + self.gamma * future
        q_values[action] += self.alpha * (target - q_values[action])

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Call once per training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path="q_table.pkl"):
        data = {
            "q_table": self.q_table,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path="q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = data["q_table"]
        return agent

    def __len__(self):
        """Number of states stored in the Q-table."""
        return len(self.q_table)
