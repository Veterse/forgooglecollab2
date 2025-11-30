import chess
import json
import os
import sys

from ai_core import SimpleAI

WEIGHTS_FILE = "chess_ai_trainable/best_weights.json"
BOARD_STATE_FILE = "chess_ai_trainable/board_state.fen"

def print_board(board):
    """Prints the board to the console with coordinates."""
    print("\n  a b c d e f g h")
    print("  -----------------")
    rows = str(board).split('\n')
    for i, row in enumerate(rows):
        print(f"{8 - i} | {row} | {8 - i}")
    print("  -----------------")
    print("  a b c d e f g h\n")

def main():
    # --- Setup ---
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Weights file not found at '{WEIGHTS_FILE}'")
        print("Please run train.py first.")
        sys.exit(1)

    with open(WEIGHTS_FILE, "r") as f:
        best_weights = json.load(f)
    ai = SimpleAI(weights=best_weights)

    # Load board state or create a new one
    if os.path.exists(BOARD_STATE_FILE):
        with open(BOARD_STATE_FILE, "r") as f:
            fen = f.read()
            board = chess.Board(fen)
    else:
        board = chess.Board()

    # --- Player's Move ---
    if len(sys.argv) < 2:
        print("Starting a new game.")
        print("You are playing as white.")
        print_board(board)
        print("To make your first move, tell me your move in UCI format (e.g., e2e4).")
        # Save initial state
        with open(BOARD_STATE_FILE, "w") as f:
            f.write(board.fen())
        sys.exit(0)

    player_move_uci = sys.argv[1]
    try:
        player_move = chess.Move.from_uci(player_move_uci)
        if player_move not in board.legal_moves:
            print(f"Invalid move: '{player_move_uci}'. That move is not legal in the current position.")
            print_board(board)
            sys.exit(1)
    except ValueError:
        print(f"Invalid move format: '{player_move_uci}'. Use UCI format (e.g., e2e4).")
        sys.exit(1)

    board.push(player_move)
    print(f"You played: {player_move_uci}")

    if board.is_game_over():
        print("--- Game Over ---")
        print(f"Result: {board.result()}")
        os.remove(BOARD_STATE_FILE) # Clean up for next game
        sys.exit(0)

    # --- AI's Move ---
    print("AI is thinking...")
    ai_move = ai.choose_move(board, depth=3)
    if ai_move:
        print(f"AI plays: {ai_move.uci()}")
        board.push(ai_move)
    else:
        # This case should be handled by is_game_over(), but as a fallback
        print("AI has no legal moves!")

    print("\nNew board state:")
    print_board(board)

    # --- Save and Exit ---
    with open(BOARD_STATE_FILE, "w") as f:
        f.write(board.fen())

    if board.is_game_over():
        print("--- Game Over ---")
        print(f"Result: {board.result()}")
        os.remove(BOARD_STATE_FILE) # Clean up for next game

if __name__ == "__main__":
    main()