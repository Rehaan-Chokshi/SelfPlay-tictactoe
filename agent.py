"""
A Q-learning agent for Tic-Tac-Toe.

The agent keeps a table mapping (board_state, action) -> expected value.
It has no built-in knowledge of strategy -- everything it "knows" comes
purely from playing games and updating values based on the outcome.
"""

import random
import pickle


class QLearningAgent:
    def __init__(self, player, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.player = player          # 1 or -1, which side this agent plays
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor for future rewards
        self.epsilon = epsilon        # exploration rate
        self.q_table = {}             # (state, action) -> value
        self.history = []             # (state, action) pairs from the current game

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, board, available_moves):
        # Explore: pick a random move
        if random.random() < self.epsilon:
            action = random.choice(available_moves)
        else:
            # Exploit: pick the move with the highest known value
            q_values = [self._get_q(board, a) for a in available_moves]
            max_q = max(q_values)
            best_moves = [a for a, q in zip(available_moves, q_values) if q == max_q]
            action = random.choice(best_moves)  # break ties randomly

        self.history.append((board, action))
        return action

    def learn_from_result(self, reward):
        """
        Called at the end of a game. Propagates the final reward backward
        through every (state, action) pair the agent took this game,
        discounting as it goes further back in time.
        """
        next_max = 0.0
        for state, action in reversed(self.history):
            old_q = self._get_q(state, action)
            new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
            self.q_table[(state, action)] = new_q
            next_max = new_q
            reward = 0  # only the final step gets the direct reward

        self.history = []  # reset for next game

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
