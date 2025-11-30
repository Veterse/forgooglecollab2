
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import numpy as np
from collections import deque
from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent
from rl_chess.RL_utils import move_to_index
import rl_chess.config as config
from rl_chess.trainer import update_network

def test_training_mechanics():
    print("--- Testing Training Mechanics (GPU, No Scaler) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy data: start position, policy target = 1.0 for e2e4, value target = 1.0
    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    state = board_to_tensor(history, device)
    
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    print(f"Index for e2e4: {idx}")
    
    policy_target = torch.zeros(4672, device=device)
    policy_target[idx] = 1.0
    value_target = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    # Batch of 10
    batch_size = 10
    state_batch = state.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    policy_batch = policy_target.unsqueeze(0).repeat(batch_size, 1)
    value_batch = value_target.unsqueeze(0).repeat(batch_size, 1)

    print("Initial prediction:")
    model.eval()
    with torch.no_grad():
        log_policy, val = model(state.unsqueeze(0))
        prob = torch.exp(log_policy[0, idx]).item()
        print(f"Prob: {prob:.6f}, Val: {val.item():.6f}")
        
    print("Training loop...")
    model.train()
    for i in range(50):
        optimizer.zero_grad()
        log_p, v = model(state_batch)
        
        v_loss = F.mse_loss(v.squeeze(), value_batch.squeeze())
        p_loss = -torch.sum(policy_batch * log_p) / batch_size
        
        loss = v_loss + p_loss
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss.item():.4f}")

    print("Post-training prediction:")
    model.eval()
    with torch.no_grad():
        log_policy, val = model(state.unsqueeze(0))
        prob = torch.exp(log_policy[0, idx]).item()
        print(f"Prob: {prob:.6f}, Val: {val.item():.6f}")
        
    if prob > 0.1:
        print("SUCCESS: Network learned on CPU.")
    else:
        print("FAILURE: Network did not learn on CPU.")

def test_mcts_noise():
    print("\n--- Testing MCTS Noise ---")
    device = torch.device("cpu") # MCTS usually on CPU for logic
    model = ChessNetwork().to(device)
    agent = MCTSAgent(model, device, num_simulations=50)
    
    board = chess.Board()
    
    # Test 1: Without noise
    print("MCTS without noise (is_self_play=False)")
    move, policy = agent.get_move(board, is_self_play=False)
    print(f"Chosen: {move}")
    # Check if policy is uniform-ish
    non_zero = policy[policy > 0]
    print(f"Max prob: {non_zero.max().item():.4f}, Min prob: {non_zero.min().item():.4f}")
    
    # Test 2: With noise
    print("MCTS with noise (is_self_play=True)")
    move, policy = agent.get_move(board, is_self_play=True)
    print(f"Chosen: {move}")
    non_zero = policy[policy > 0]
    print(f"Max prob: {non_zero.max().item():.4f}, Min prob: {non_zero.min().item():.4f}")
    
    if non_zero.max().item() > 0.1:
        print("SUCCESS: MCTS with noise produced spiky policy.")
    else:
        print("FAILURE: MCTS with noise produced uniform policy.")

if __name__ == "__main__":
    test_training_mechanics()
    test_mcts_noise()
