"""
Play against the trained agent in the terminal.

Run train.py first to generate agent_x.pkl / agent_o.pkl.
You play as O; the trained agent plays as X and moves first.
"""

from game import new_board, available_moves, apply_move, check_winner, print_board
from agent import QLearningAgent


def human_move(board):
    moves = available_moves(board)
    while True:
        try:
            choice = int(input(f"Your move {moves}: "))
            if choice in moves:
                return choice
        except ValueError:
            pass
        print("Invalid move, try again.")


def play():
    agent_x = QLearningAgent(player=1, epsilon=0.0)  # no exploration, best moves only
    agent_x.load("agent_x.pkl")

    board = new_board()
    current_player = 1

    print("You are O. The trained agent is X and moves first.\n")
    print_board(board)
    print()

    while True:
        if current_player == 1:
            moves = available_moves(board)
            action = agent_x.choose_action(board, moves)
            print(f"Agent plays position {action}")
        else:
            action = human_move(board)

        board = apply_move(board, action, current_player)
        print_board(board)
        print()

        result = check_winner(board)
        if result is not None:
            if result == 1:
                print("Agent (X) wins!")
            elif result == -1:
                print("You (O) win!")
            else:
                print("It's a draw.")
            break

        current_player *= -1


if __name__ == "__main__":
    play()
