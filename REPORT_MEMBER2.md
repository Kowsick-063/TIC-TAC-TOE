# Member 2 - Report Sections
*Reinforcement Learning theory, MDP formulation, Q-Learning equation and algorithm*

## 1. Reinforcement Learning Theory
Reinforcement Learning (RL) is a branch of machine learning in which an **agent** learns to make decisions by interacting with an **environment**. At each time step the agent observes a state, chooses an action, and receives a numerical reward and a new state. The agent is never told the correct move; it must discover good behaviour by trial and error, with the goal of maximising the cumulative (discounted) reward over time. The behaviour it learns is called a **policy**.

Key ideas used in this project:
- **Value function / action-value function** - Q(s, a) estimates how good it is to take action *a* in state *s* and then act well afterwards.
- **Temporal Difference (TD) learning** - estimates are updated after every step using the difference between the current estimate and a better one built from the observed reward plus the estimate of the next state (bootstrapping). No complete game needs to be finished before learning.
- **Model-free, off-policy learning** - Q-Learning does not need the rules/transition model of the game, and it learns the value of the greedy policy while still following an exploratory one.
- **Exploration vs exploitation** - the agent must try new moves (explore) as well as use what it already knows (exploit).

## 2. MDP Formulation of Tic-Tac-Toe
The game is modelled as a Markov Decision Process (S, A, P, R, gamma):

| Component | Definition in this project |
|---|---|
| **State (S)** | The 3x3 board as a tuple of 9 values: 0 empty, +1 the agent's mark, -1 the opponent's mark (agent's perspective). Whose turn it is is implicit because the agent only decides on its own turn. |
| **Action (A)** | Choose an empty cell, 0-8. The legal set A(s) shrinks as the board fills. |
| **Transition (P)** | The agent's move followed by the opponent's reply (random, self-play or other). The next state is the board where the agent moves again, so the opponent is part of the environment. |
| **Reward (R)** | Win +1, draw +0.2, loss -1, game continues 0. |
| **Discount (gamma)** | 0.9 - rewards that are nearer count slightly more, so faster wins are preferred. |
| **Terminal states** | A line of three or a full board. Their future value is 0. |

The Markov property holds: the current board contains everything needed to decide the next move, regardless of how it was reached.

## 3. Q-Learning Equation
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
\]

| Symbol | Meaning | Value used |
|---|---|---|
| Q(s,a) | current action-value estimate | initialised to 0 |
| alpha | learning rate | 0.5 |
| r | immediate reward | +1 / 0.2 / -1 / 0 |
| gamma | discount factor | 0.9 |
| s, a | current state and chosen action | |
| s', a' | next state and its legal actions | max taken over **legal** a' only |

The term in brackets is the **TD error**: the gap between the new target (r + gamma max Q(s',a')) and the old estimate. For a terminal s' the target is just r.

The state-space size is small (a few thousand reachable boards), so a plain Python dictionary is used as the Q-table: `state -> [Q for cells 0..8]`.

## 4. Algorithm
**Epsilon-greedy action selection**
```
with probability epsilon : choose a random legal action        (explore)
otherwise                : choose the legal action with max Q  (exploit, ties random)
```
epsilon starts at 1.0 and is multiplied by 0.9995 after every episode until it reaches 0.05, so early games are exploratory and later games rely on learned values.

**Q-Learning with a two-player game**
```
initialise Q(s,a) = 0 for all s, a
repeat for each episode:
    reset the board; pending = None
    while the game is not over:
        if it is the agent's turn:
            s = current board
            if pending exists:  Q-update(pending, reward 0, next state s)
            a = epsilon-greedy(s);  pending = (s, a);  play a
        else:
            opponent plays
        if the game is over:
            r = +1 win / 0.2 draw / -1 loss
            Q-update(pending, r, terminal)       # future term = 0
    decay epsilon
```

**After training** the agent plays greedily (epsilon = 0) so that evaluation games and human games use only the learned policy and do not modify the Q-table.

## 5. Verification Results (Member 2 module test)
A short 30,000-episode run against a random opponent (agent alternating X and O), then 1,000 greedy evaluation games vs a random opponent:

| Metric | Before training | After training |
|---|---|---|
| Wins | 435 | 912 |
| Losses | 442 | 5 |
| Draws | 123 | 83 |
| Win rate | 43.5 % | 91.2 % |
| States in Q-table | 0 | 3,882 |

*(Results vary slightly by random seed; Member 3's full evaluation supersedes this table in the final report.)*
