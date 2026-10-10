"""Tests for agent.py (Member 2). Run:  python test_agent.py"""

import os
import random
import tempfile

from game import TicTacToe, X, O
from agent import (QLearningAgent, REWARD_WIN, REWARD_DRAW, REWARD_LOSS)

# ---------------------------------------------------------------- unit tests

# 1. State is stored from the agent's perspective.
board = (1, 0, 0, 0, -1, 0, 0, 0, 0)
assert QLearningAgent.encode_state(board, X) == board
assert QLearningAgent.encode_state(board, O) == (-1, 0, 0, 0, 1, 0, 0, 0, 0)
print("Test 1 passed: state encoding uses the agent's perspective.")

# 2. Rewards.
assert QLearningAgent.get_reward(X, X) == REWARD_WIN == 1
assert QLearningAgent.get_reward(O, X) == REWARD_LOSS == -1
assert QLearningAgent.get_reward("draw", O) == REWARD_DRAW == 0.2
assert QLearningAgent.get_reward(None, O) == 0
print("Test 2 passed: rewards for win / loss / draw / ongoing.")

# 2b. The agent can identify legal actions from a state on its own.
assert QLearningAgent.get_legal_actions((1, 0, 0, 0, -1, 0, 1, 0, 0)) == [1, 2, 3, 5, 7, 8]
assert QLearningAgent.get_legal_actions((1, -1, 1, -1, 1, -1, -1, 1, -1)) == []
assert QLearningAgent().choose_action((1, 1, -1, -1, -1, 1, 1, 0, 1), None, X, False) == 7
print("Test 2b passed: agent finds legal actions from the board itself.")

# 3. Only legal actions are chosen (exploring and exploiting).
agent = QLearningAgent(epsilon=1.0)
b = (1, -1, 1, 0, -1, 0, 0, 0, 0)
legal = [3, 5, 6, 7, 8]
for _ in range(300):
    assert agent.choose_action(b, legal, O, training=True) in legal
agent.epsilon = 0.0
agent.q_table[agent.encode_state(b, O)] = [9, 9, 9, 0.1, 0, 0.5, 0, 0, 0]  # illegal cells hold big values
for _ in range(50):
    assert agent.choose_action(b, legal, O, training=True) == 5
print("Test 3 passed: agent only picks legal moves (greedy picks best legal).")

# 4. Q-Learning update, non-terminal. alpha=0.5, gamma=0.9
agent = QLearningAgent(alpha=0.5, gamma=0.9)
s = (0,) * 9
s2 = (1, 0, 0, 0, -1, 0, 0, 0, 0)
agent.q_table[agent.encode_state(s2, X)] = [0.9, 0.4, 0.8, 0, 0, 0, 0, 0, 0]  # cell 0 is occupied -> 0.9 must be ignored
agent.update(s, 4, 0.0, s2, [1, 2, 3, 5, 6, 7, 8], False, X)
# max over LEGAL actions only = 0.8 (occupied cell 0 holds 0.9)
expected = 0 + 0.5 * (0 + 0.9 * 0.8 - 0)
assert abs(agent.get_q(s, 4, X) - expected) < 1e-9
print("Test 4 passed: Q-update equation (max taken over legal actions).")

# 5. Terminal update ignores future value.
agent = QLearningAgent(alpha=0.5, gamma=0.9)
final = (1, 1, 1, -1, -1, 0, 0, 0, 0)
agent.q_table[agent.encode_state(final, X)] = [5] * 9   # must be ignored
agent.update(s, 2, REWARD_WIN, final, [], True, X)
assert abs(agent.get_q(s, 2, X) - 0.5) < 1e-9
print("Test 5 passed: terminal states use only the reward.")

# 6. Epsilon decay respects the floor.
agent = QLearningAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
for _ in range(10):
    agent.decay_epsilon()
assert agent.epsilon == 0.1
print("Test 6 passed: epsilon decays down to epsilon_min.")

# 7. Save / load round trip.
agent = QLearningAgent(alpha=0.3)
agent.update(s, 0, 1.0, final, [], True, X)
path = os.path.join(tempfile.mkdtemp(), "q.pkl")
agent.save(path)
loaded = QLearningAgent.load(path)
assert loaded.q_table == agent.q_table and loaded.alpha == 0.3
print("Test 7 passed: Q-table saves and loads.")

# ------------------------------------------- integration / learning sanity test
# (A minimal loop; Member 3's train.py is the full version.)

def play_episode(agent, me, opponent_policy, training=True):
    game = TicTacToe()
    pending = None  # (board, action) waiting for its next state

    while not game.done:
        if game.current_player == me:
            board, legal = game.get_state(), game.get_legal_actions()
            if training and pending:
                agent.update(pending[0], pending[1], 0.0, board, legal, False, me)
            action = agent.choose_action(board, legal, me, training)
            pending = (board, action)
            game.make_move(action)
        else:
            game.make_move(opponent_policy(game))

        if game.done and training and pending:
            r = agent.get_reward(game.get_result(), me)
            agent.update(pending[0], pending[1], r, game.get_state(), [], True, me)

    return game.get_result()


def random_policy(game):
    return random.choice(game.get_legal_actions())


def evaluate(agent, n=1000):
    out = {"win": 0, "loss": 0, "draw": 0}
    for i in range(n):
        me = X if i % 2 == 0 else O
        res = play_episode(agent, me, random_policy, training=False)
        out["draw" if res == "draw" else "win" if res == me else "loss"] += 1
    return out


random.seed(1)
agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=1.0,
                       epsilon_min=0.05, epsilon_decay=0.9995)
before = evaluate(agent)

for episode in range(30000):
    me = X if episode % 2 == 0 else O
    play_episode(agent, me, random_policy, training=True)
    agent.decay_epsilon()

after = evaluate(agent)
print(f"Before training vs random: {before}")
print(f"After  training vs random: {after}   (Q-table states: {len(agent)})")
assert after["win"] > before["win"]
assert after["win"] >= 900, "agent should beat a random player ~90%+"
print("Test 8 passed: agent learns to beat a random opponent.")
