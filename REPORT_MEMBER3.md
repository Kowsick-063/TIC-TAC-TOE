# TIC-TAC-TOE USING Q-LEARNING
## Member 3 – Training, Evaluation, and Integration Report

**Project Title:** Tic-Tac-Toe AI Using Q-Learning  
**Contribution:** Model Training, Performance Evaluation, Model Persistence, and GUI Integration  
**Repository:** `E:\TIC-TAC-TOE`

---

## 1. Introduction

The Tic-Tac-Toe Q-Learning project aims to develop an intelligent game-playing agent that learns to make decisions through reinforcement learning. Instead of relying entirely on predefined rules, the agent learns from repeated game experiences and improves its decision-making strategy over time.

My contribution as Member 3 focuses on training the Q-Learning agent, evaluating its performance against different opponents, saving the trained model, generating performance visualizations, and integrating the trained agent with the graphical user interface.

All the relevant implementation, model files, evaluation results, and GUI components are maintained in the team's shared repository. The project has been tested and executed successfully from this repository.

## 2. Objectives

The primary objectives of my contribution are:

- Train the Q-Learning agent through repeated game episodes.
- Evaluate the agent against random and optimal opponents.
- Save and reload the trained Q-table for reuse.
- Generate training logs, evaluation results, and performance graphs.
- Integrate the trained agent into the graphical user interface.
- Verify the functionality of the complete application through automated and manual testing.

## 3. Methodology

### 3.1 Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking an action in a particular state. In this project, the agent observes the current Tic-Tac-Toe board, selects a legal move, receives a reward based on the outcome, and updates its learned action values.

The Q-value update equation is:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\]

Where:

- \(s\) represents the current board state.
- \(a\) represents the selected move.
- \(r\) represents the reward received.
- \(s'\) represents the next state.
- \(\alpha\) is the learning rate.
- \(\gamma\) is the discount factor.

### 3.2 Exploration and Exploitation

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

- **Exploration:** The agent tries different legal moves to discover potentially better strategies.
- **Exploitation:** The agent selects moves with higher learned Q-values.

During training, the exploration rate decreases so that the agent gradually relies more on its learned strategy.

### 3.3 Training Process

The training pipeline was executed for **60,000 episodes**. After training, the learned Q-table contained **4,477 states**, and the saved agent had an exploration rate of **0.05**.

The resulting model was stored in `q_table.pkl` so that the application could reuse the learned policy without retraining every time the game starts.

## 4. Training and Evaluation

The trained agent was evaluated against two types of opponents.

### 4.1 Evaluation Against a Random Opponent

A random opponent selects legal moves without following a learned strategy. This evaluation helps determine whether the trained agent has learned to make better decisions than an opponent playing randomly.

| Metric | Result |
|---|---:|
| Total games | 1,000 |
| Wins | 920 |
| Losses | 0 |
| Draws | 80 |
| Win rate | 92% |

The results indicate that the trained agent performed strongly against the random opponent, winning most games and avoiding losses in the recorded evaluation.

### 4.2 Evaluation Against an Optimal Opponent

An optimal opponent selects moves to avoid losing and exploit mistakes. Since Tic-Tac-Toe permits a draw when both players play optimally, avoiding defeat is an important performance measure.

| Metric | Result |
|---|---:|
| Total games | 1,000 |
| Wins | 0 |
| Losses | 0 |
| Draws | 1,000 |
| Draw rate | 100% |

The agent drew all 1,000 recorded games against the optimal opponent. This indicates that it successfully avoided losses in this evaluation. The result is consistent with optimal Tic-Tac-Toe play, where a draw is the expected outcome when neither player makes a mistake.

## 5. Training Visualizations

The training pipeline generated visualizations to support analysis of the agent's learning process.

### 5.1 Learning Curve — `learning_curve.png`

The learning curve provides a visual representation of the agent's performance during training. It helps assess whether the agent's policy improves as it experiences more game episodes.

### 5.2 Epsilon Curve — `epsilon_curve.png`

The epsilon curve illustrates the change in the exploration rate during training. A decreasing exploration rate allows the agent to transition gradually from trying different moves to using its learned strategy more frequently.

### 5.3 Outcome Visualization — `outcomes.png`

The outcome graph summarizes game results and provides a visual comparison of wins, losses, and draws recorded during evaluation.

The generated visualizations can be used in the final project presentation to explain the learning process and demonstrate the evaluation results.

## 6. Model Saving and Loading

After training, the learned Q-table and relevant agent parameters were saved in `q_table.pkl`.

The saved model was subsequently loaded using the agent's loading functionality. Verification confirmed that the loaded Q-table contained 4,477 states, demonstrating that the learned state information could be restored successfully.

This approach separates training from gameplay. The application can load the previously trained model instead of repeating the entire training process whenever the user opens the game.

## 7. GUI Integration

The trained agent was integrated into the Tic-Tac-Toe graphical user interface.

The application loads the saved model through the game interface's model-loading configuration. The GUI allows a user to play Tic-Tac-Toe against the trained AI.

Manual testing was performed by launching the GUI and playing multiple games against the AI. The interface and gameplay were observed to function correctly during testing.

The integration connects the main components of the project:

1. The game logic manages the Tic-Tac-Toe board and legal moves.
2. The Q-Learning agent selects moves based on its learned values.
3. The saved Q-table supplies the trained policy.
4. The GUI provides the interface through which the user plays the game.

## 8. Testing and Verification

The following checks were completed:

- The automated test suite was executed, and the tests passed.
- The training pipeline completed successfully.
- The trained Q-table was saved to disk.
- The saved model was loaded successfully, and its state count was verified.
- Evaluation results and training visualizations were generated.
- The GUI was launched and manually tested through multiple games.
- The complete application was tested from the team's shared repository.

These checks support the successful operation of the training, model-loading, evaluation, and gameplay components.

## 9. Important Project Files

| File | Purpose |
|---|---|
| `agent.py` | Implements the Q-Learning agent and model persistence functionality. |
| `game.py` | Contains the Tic-Tac-Toe game logic. |
| `train.py` | Executes training and evaluation routines. |
| `main.py` | Launches the graphical interface and connects it to the trained agent. |
| `q_table.pkl` | Stores the trained Q-table and agent parameters. |
| `training_log.csv` | Records training information. |
| `results.csv` | Stores evaluation results. |
| `learning_curve.png` | Visualizes learning performance. |
| `epsilon_curve.png` | Visualizes the exploration rate. |
| `outcomes.png` | Summarizes game outcomes. |

## 10. Challenges and Solutions

### Challenge 1: Learning an Effective Strategy

An agent must explore different moves before it can learn which actions are effective.

**Solution:** The epsilon-greedy strategy was used to balance exploration and exploitation during training.

### Challenge 2: Reusing the Trained Model

Training the agent whenever the game starts would increase startup time and make the application less convenient to use.

**Solution:** The learned Q-table was saved to a file and successfully reloaded for gameplay.

### Challenge 3: Evaluating the Agent's Performance

A high win rate against a weak opponent alone is insufficient to determine whether an agent can avoid defeat against stronger play.

**Solution:** The agent was evaluated against both a random opponent and an optimal opponent to examine its performance under different conditions.

### Challenge 4: Connecting the Model to the Interface

The trained agent needed to work with the game's board representation and legal-move handling.

**Solution:** The saved agent was connected to the existing game interface, and gameplay was manually tested.

## 11. Conclusion

My contribution to the Tic-Tac-Toe Q-Learning project involved training the reinforcement learning agent, evaluating its performance, verifying model persistence, generating supporting visualizations, and integrating the trained model with the graphical user interface.

The agent was trained for 60,000 episodes and learned a Q-table containing 4,477 states. During evaluation, it achieved 920 wins and 80 draws in 1,000 games against a random opponent, with no recorded losses. Against an optimal opponent, it drew all 1,000 games without losing.

The model was saved and reloaded successfully, the automated tests passed, and the GUI was tested from the shared repository. These results demonstrate the successful integration of the learning component with the playable Tic-Tac-Toe application.

## 12. Suggested Demonstration Sequence

1. Open the project from the shared repository.
2. Show the training script and explain the Q-Learning update process.
3. Present the saved model, `q_table.pkl`.
4. Display the learning curve, epsilon curve, and outcome visualization.
5. Explain the evaluation results against random and optimal opponents.
6. Launch `main.py` and demonstrate a game against the trained AI.
7. Briefly explain how the GUI loads and uses the trained model.

---

**Note before submission:** Add your name, team number, institution details, and any required cover-page information. Insert the actual plots and screenshots from the repository in the relevant sections. Confirm that the evaluation figures match the final `results.csv` before submitting the report.
