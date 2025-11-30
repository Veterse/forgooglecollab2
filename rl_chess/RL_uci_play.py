import chess
import chess.engine
import torch
import os
import sys
import logging
from collections import deque

# Reconfigure the thinking logger for UCI play to use a separate file.
# This avoids mixing logs with the training script.
think_logger = logging.getLogger("thinking")
if think_logger.hasHandlers():
    think_logger.handlers.clear()
fh = logging.FileHandler("uci_thinking_log.txt", mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(message)s"))
think_logger.addHandler(fh)
think_logger.setLevel(logging.DEBUG)

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent
from rl_chess import config

# Simple logging to a file for debugging UCI communication
def log(message):
    with open("uci_log.txt", "a") as f:
        f.write(message + "\n")

class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = deque([self.board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
        self.load_model()

    def load_model(self):
        try:
            log(f"Current Working Directory: {os.getcwd()}")
            # The working directory is set to the project root by the GUI,
            # so we can directly look for the model file.
            model_path = "rl_chess_model.pth"
            
            if os.path.exists(model_path):
                log("Loading model from: " + os.path.abspath(model_path))
                
                # Load the state dictionary
                state_dict = torch.load(model_path, map_location=self.device)

                # Fix keys if the model was saved with torch.compile()
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    log("Compiled model detected, stripping prefixes...")
                    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
                    state_dict = new_state_dict

                self.agent = MCTSAgent(ChessNetwork().to(self.device), self.device)
                self.agent.model.load_state_dict(state_dict)
                self.agent.model.eval()
                log("Model loaded successfully.")
            else:
                log(f"Model file not found at {os.path.abspath(model_path)}. The engine will not work.")
                self.agent = None
        except Exception as e:
            log(f"Error loading model: {e}")
            self.agent = None

    def uci_loop(self):
        log("UCI Engine started.")
        while True:
            line = sys.stdin.readline().strip()
            log(f"Received: {line}")
            if not line:
                continue

            parts = line.split()
            command = parts[0]

            if command == "uci":
                self.handle_uci()
            elif command == "isready":
                self.handle_isready()
            elif command == "ucinewgame":
                self.handle_ucinewgame()
            elif command == "position":
                self.handle_position(parts[1:])
            elif command == "go":
                self.handle_go(parts[1:])
            elif command == "stop":
                # For now, we don't support stopping mid-thought, as our search is fast.
                pass
            elif command == "quit":
                log("Quitting.")
                break

    def handle_uci(self):
        print("id name RL_Chess_Engine")
        print("id author YourName") # You can change this
        print("uciok")
        sys.stdout.flush()

    def handle_isready(self):
        # We can add checks here if model loading takes time. For now, it's instant.
        print("readyok")
        sys.stdout.flush()

    def handle_ucinewgame(self):
        self.board.reset()
        self.history = deque([self.board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
        # Optional: reload the model for the new game to get the latest version
        self.load_model()

    def handle_position(self, parts):
        if parts[0] == "startpos":
            self.board.reset()
            self.history = deque([self.board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
            if len(parts) > 1 and parts[1] == "moves":
                for move in parts[2:]:
                    self.board.push_uci(move)
                    self.history.append(self.board.copy())
        elif parts[0] == "fen":
            fen_parts = parts[1:]
            moves_index = -1
            try:
                moves_index = fen_parts.index("moves")
            except ValueError:
                pass

            if moves_index != -1:
                fen = " ".join(fen_parts[:moves_index])
                moves = fen_parts[moves_index+1:]
            else:
                fen = " ".join(fen_parts)
                moves = []
            
            self.board.set_fen(fen)
            self.history = deque([self.board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
            for move in moves:
                self.board.push_uci(move)
                self.history.append(self.board.copy())

    def handle_go(self, parts):
        if not self.agent:
            log("No model loaded, cannot process 'go' command.")
            print("bestmove 0000") # Null move
            sys.stdout.flush()
            return
            
        # Our agent doesn't use time controls yet, it uses a fixed number of simulations.
        # The number of simulations is now set in the MCTSAgent constructor.
        best_move, _ = self.agent.get_move(self.board, board_history=self.history)
        
        if best_move:
            log(f"Found best move: {best_move.uci()}")
            print(f"bestmove {best_move.uci()}")
        else:
            # This can happen if the game is over
            log("No legal moves found.")
            print("bestmove 0000") # Null move
            
        sys.stdout.flush()


if __name__ == "__main__":
    engine = UCIEngine()
    engine.uci_loop() 