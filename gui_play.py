
import pygame
import chess
import chess.svg
import sys
import os
import json
from _io import BytesIO

from ai_core import SimpleAI

# --- Constants ---
SCREEN_SIZE = 600
BOARD_SIZE = 560 # To leave some margin
MARGIN = (SCREEN_SIZE - BOARD_SIZE) // 2
SQUARE_SIZE = BOARD_SIZE // 8
WEIGHTS_FILE = "chess_ai_trainable/best_weights.json"

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Gemini Chess AI")

# --- Asset Loading ---

def load_board_texture(board, selected_square):
    """Draws the board and pieces as text. A fallback for when SVG rendering fails."""
    board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
    font = pygame.font.SysFont(None, 48)
    coord_font = pygame.font.SysFont(None, 24)

    for r in range(8):
        for f in range(8):
            square_idx = chess.square(f, 7 - r)
            color = (238, 238, 210) if (r + f) % 2 == 0 else (118, 150, 86) # White/Green squares
            
            # Highlight selected square and legal moves
            if selected_square is not None:
                if square_idx == selected_square:
                    color = (165, 42, 42) # Brown for selected
                else:
                    for move in board.legal_moves:
                        if move.from_square == selected_square and move.to_square == square_idx:
                            color = (0, 255, 0) # Green for legal moves

            pygame.draw.rect(board_surface, color, (f * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Draw piece symbols
            piece = board.piece_at(square_idx)
            if piece:
                piece_symbol = piece.symbol()
                text_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                text_surface = font.render(piece_symbol, True, text_color)
                text_rect = text_surface.get_rect(center=(f * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2))
                board_surface.blit(text_surface, text_rect)

            # Draw coordinates
            if f == 0:
                coord_text = coord_font.render(str(8 - r), True, (200, 200, 200))
                board_surface.blit(coord_text, (f * SQUARE_SIZE + 2, r * SQUARE_SIZE + 2))
            if r == 7:
                coord_text = coord_font.render(chess.FILE_NAMES[f], True, (200, 200, 200))
                board_surface.blit(coord_text, (f * SQUARE_SIZE + SQUARE_SIZE - 12, r * SQUARE_SIZE + SQUARE_SIZE - 20))

    return board_surface


def get_square_from_mouse(pos):
    """Converts mouse coordinates to a chess square index."""
    x, y = pos
    if not (MARGIN < x < SCREEN_SIZE - MARGIN and MARGIN < y < SCREEN_SIZE - MARGIN):
        return None
    file_idx = (x - MARGIN) // SQUARE_SIZE
    rank_idx = 7 - ((y - MARGIN) // SQUARE_SIZE) # Y is inverted in pygame
    return chess.square(file_idx, rank_idx)

def main():
    # --- Load AI ---
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: Weights file not found at '{WEIGHTS_FILE}'. Please run train.py first.")
        sys.exit(1)
    with open(WEIGHTS_FILE, "r") as f:
        best_weights = json.load(f)
    ai = SimpleAI(weights=best_weights)

    # --- Game State ---
    board = chess.Board()
    selected_square = None
    player_turn = True # Player is always white

    # --- Game Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and player_turn and not board.is_game_over():
                clicked_square = get_square_from_mouse(event.pos)
                if clicked_square is not None:
                    # If a square was already selected, try to make a move
                    if selected_square is not None:
                        move = chess.Move(selected_square, clicked_square)
                        # Check for promotion
                        if board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(clicked_square) == 7:
                            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                            player_turn = False # Switch to AI's turn
                        else:
                            # If the click is on another of our pieces, select it instead
                            if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                                selected_square = clicked_square
                            else:
                                selected_square = None # Deselect
                    # If no square was selected, select the clicked one if it's our piece
                    elif board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                        selected_square = clicked_square

        # --- AI's Turn ---
        if not player_turn and not board.is_game_over():
            pygame.display.set_caption("Gemini Chess AI - AI is thinking...")
            ai_move = ai.choose_move(board, depth=3)
            if ai_move:
                board.push(ai_move)
            player_turn = True # Switch back to player's turn
            pygame.display.set_caption("Gemini Chess AI - Your turn")

        # --- Drawing ---
        screen.fill((30, 30, 30)) # Dark grey background
        board_surface = load_board_texture(board, selected_square)
        screen.blit(board_surface, (MARGIN, MARGIN))
        
        if board.is_game_over():
            font = pygame.font.SysFont(None, 50)
            result_text = f"Game Over: {board.result()}"
            text_surface = font.render(result_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(SCREEN_SIZE/2, SCREEN_SIZE/2))
            pygame.draw.rect(screen, (0, 0, 0, 150), text_rect.inflate(20, 20)) # semi-transparent background
            screen.blit(text_surface, text_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
