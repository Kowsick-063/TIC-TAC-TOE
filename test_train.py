"""Tests for train.py (Member 3). Run:  python test_train.py"""
import os
import random

from game import X, O
from agent import QLearningAgent
from train import (play_episode, evaluate, random_policy, optimal_policy,
                   train)

random.seed(0)

# 1. The perfect opponent never loses to itself (always a draw).
agent = QLearningAgent()
for _ in range(50):
    result, _ = play_episode(agent, [], optimal_policy, training=False)
    assert result == "draw"
print("Test 1 passed: minimax opponent plays perfectly.")

# 2. Evaluation counts add up and training fills the Q-table.
agent = QLearningAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999)
stats = evaluate(agent, random_policy, 200)
assert stats["wins"] + stats["losses"] + stats["draws"] == 200
log = train(agent, 3000, "mixed", 1000, 100)
assert len(agent) > 500 and len(log) == 3
print("Test 2 passed: training runs and the Q-table grows.")

# 3. Evaluation does not change the Q-table.
snapshot = {k: list(v) for k, v in agent.q_table.items()}
evaluate(agent, random_policy, 100)
assert all(agent.q_table[k] == v for k, v in snapshot.items())
print("Test 3 passed: evaluation never modifies the Q-table.")

# 4. The saved model (q_table.pkl) never loses to a perfect player.
if os.path.exists("q_table.pkl"):
    trained = QLearningAgent.load("q_table.pkl")
    res = evaluate(trained, optimal_policy, 300)
    assert res["losses"] == 0, res
    res = evaluate(trained, random_policy, 300)
    assert res["win_rate"] > 80, res
    print("Test 4 passed: trained agent never loses to optimal play "
          f"and wins {res['win_rate']:.0f}% vs random.")
else:
    print("Test 4 skipped: run `python train.py` first.")
