"""
Trains a Tic-Tac-Toe agent purely through self-play.

Two instances of the same agent architecture play against each other
thousands of times. Each game's outcome (win / loss / draw) is used to
update both agents' Q-tables. Win rate is tracked in "generations" of
games so you can watch the agent improve over time.
"""

import matplotlib.pyplot as plt

from game import new_board, available_moves, apply_move, check_winner
from agent import QLearningAgent

NUM_GAMES = 50_000
GENERATION_SIZE = 500   # games per reported win-rate bucket


def play_one_game(agent_x, agent_o):
    board = new_board()
    current_player = 1

    while True:
        agent = agent_x if current_player == 1 else agent_o
        moves = available_moves(board)
        action = agent.choose_action(board, moves)
        board = apply_move(board, action, current_player)

        result = check_winner(board)
        if result is not None:
            return result  # 1 = X wins, -1 = O wins, 0 = draw

        current_player *= -1


def reward_for(result, player):
    if result == player:
        return 1.0
    if result == 0:
        return 0.3   # small reward for a draw -- better than losing
    return -1.0


def train():
    agent_x = QLearningAgent(player=1)
    agent_o = QLearningAgent(player=-1)

    x_win_rate_history = []
    draw_rate_history = []

    wins_x = wins_o = draws = 0

    for game_num in range(1, NUM_GAMES + 1):
        # decay exploration over time so the agent exploits more as it learns
        progress = game_num / NUM_GAMES
        agent_x.epsilon = max(0.05, 0.3 * (1 - progress))
        agent_o.epsilon = max(0.05, 0.3 * (1 - progress))

        result = play_one_game(agent_x, agent_o)

        agent_x.learn_from_result(reward_for(result, 1))
        agent_o.learn_from_result(reward_for(result, -1))

        if result == 1:
            wins_x += 1
        elif result == -1:
            wins_o += 1
        else:
            draws += 1

        if game_num % GENERATION_SIZE == 0:
            x_win_rate_history.append(wins_x / GENERATION_SIZE)
            draw_rate_history.append(draws / GENERATION_SIZE)
            print(
                f"Games {game_num:>6} | "
                f"X wins: {wins_x:>4} | O wins: {wins_o:>4} | Draws: {draws:>4}"
            )
            wins_x = wins_o = draws = 0

    agent_x.save("agent_x.pkl")
    agent_o.save("agent_o.pkl")
    print("\nSaved trained agents to agent_x.pkl and agent_o.pkl")

    plot_progress(x_win_rate_history, draw_rate_history)


def plot_progress(x_win_rate_history, draw_rate_history):
    generations = range(1, len(x_win_rate_history) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(generations, x_win_rate_history, label="X win rate", linewidth=2)
    plt.plot(generations, draw_rate_history, label="Draw rate", linewidth=2)
    plt.xlabel(f"Generation ({GENERATION_SIZE} games each)")
    plt.ylabel("Rate")
    plt.title("Self-play Tic-Tac-Toe: win/draw rate over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    print("Saved training_progress.png")


if __name__ == "__main__":
    train()
