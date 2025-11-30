# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from . import config

# More neurons for higher strength (requires serious resources)
NUM_RESIDUAL_BLOCKS = 10 #тоже уменьшил,не помню почему,но пусть будет для скорости тогда
NUM_CHANNELS = 175  # was 256 - number of filters in conv layers временно уменьшим

class ResidualBlock(nn.Module):
    """
    Residual block, main building unit of the network.
    Consists of two convolutional layers and skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Key element - skip connection
        out = F.relu(out)
        return out

class ChessNetwork(nn.Module):
    """
    Main neural network inspired by AlphaZero architecture.
    """
    def __init__(self):
        super(ChessNetwork, self).__init__()
        # 1. Input layer: converts history-aware board representation to NUM_CHANNELS
        self.conv_input = nn.Conv2d(config.INPUT_CHANNELS, NUM_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(NUM_CHANNELS)

        # 2. Network body: residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(NUM_CHANNELS, NUM_CHANNELS) for _ in range(NUM_RESIDUAL_BLOCKS)])

        # 3. Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # 4672 - all possible chess moves (including pawn promotions)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

        # 4. Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Pass data through input layer
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # Pass through network body
        for block in self.residual_blocks:
            x = block(x)

        # Policy head output
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)

        # Value head output
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(-1, 1 * 8 * 8)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        
        # Return move probabilities (after log_softmax) and position evaluation (after tanh)
        return F.log_softmax(policy, dim=1), torch.tanh(value)

def board_to_tensor(board_history, device):
    """
    Converts a history of chess.Board positions into a tensor of shape
    (config.INPUT_CHANNELS, 8, 8) for neural network input.

    History consists of BOARD_HISTORY_LENGTH snapshots (oldest -> newest).
    Each snapshot contributes 12 channels (6 white piece types + 6 black piece types).
    The last 6 channels store auxiliary info (side to move, halfmove clock, castling rights)
    for the most recent board.
    """

    if isinstance(board_history, chess.Board):
        history_sequence = [board_history]
    else:
        history_sequence = list(board_history)

    if not history_sequence:
        raise ValueError("board_to_tensor expects at least one board in history")

    max_len = config.BOARD_HISTORY_LENGTH
    tensor = torch.zeros((config.INPUT_CHANNELS, 8, 8), dtype=torch.float32)

    # Pad history on the left with None for missing entries, keep latest positions
    trimmed_history = history_sequence[-max_len:]
    padding = [None] * (max_len - len(trimmed_history))
    ordered_history = padding + trimmed_history

    for idx, board in enumerate(ordered_history):
        if board is None:
            continue

        base_channel = idx * config.PIECE_CHANNELS_PER_POSITION
        for color in chess.COLORS:
            color_offset = 0 if color == chess.WHITE else 6
            for piece_type in chess.PIECE_TYPES:
                channel_index = base_channel + color_offset + (piece_type - 1)
                for square in board.pieces(piece_type, color):
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    tensor[channel_index, rank, file] = 1.0

    latest_board = ordered_history[-1]
    aux_start = config.BOARD_HISTORY_LENGTH * config.PIECE_CHANNELS_PER_POSITION

    # Current player color (1 if white to move, else 0)
    tensor[aux_start, :, :] = 1.0 if latest_board.turn == chess.WHITE else 0.0
    # Halfmove clock (normalized)
    tensor[aux_start + 1, :, :] = latest_board.halfmove_clock / 100.0
    # Castling rights channels
    tensor[aux_start + 2, :, :] = 1.0 if latest_board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[aux_start + 3, :, :] = 1.0 if latest_board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[aux_start + 4, :, :] = 1.0 if latest_board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[aux_start + 5, :, :] = 1.0 if latest_board.has_queenside_castling_rights(chess.BLACK) else 0.0

    return tensor.to(device)