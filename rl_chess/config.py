# -*- coding: utf-8 -*-
"""
Единый файл конфигурации для всех компонентов системы.
"""

import torch

# --- Представление состояния ---
BOARD_HISTORY_LENGTH = 8  # Сколько последних позиций подаем сети
PIECE_CHANNELS_PER_POSITION = 12  # 6 типов фигур на цвет
AUXILIARY_CHANNELS = 6  # Чей ход, halfmove clock, рокировки
INPUT_CHANNELS = BOARD_HISTORY_LENGTH * PIECE_CHANNELS_PER_POSITION + AUXILIARY_CHANNELS

# --- Параметры MCTS ---
# ФАЗА 1: Быстрая генерация (0-5000 игр) - скорость важнее качества
# После 5000 игр увеличить до 400-800
MCTS_SIMULATIONS = 150  # Уменьшено для быстрой генерации данных на старте
MCTS_BATCH_SIZE = 16    # Сколько листовых узлов MCTS обрабатывать за один проход нейросети

# --- Параметры обучения ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 256
BATCH_SIZE = 256  # Размер батча для обучения (sequential mode)
SCHEDULER_GAMMA = 1.0 # Коэффициент затухания отключен для стабильного старта
EPOCHS_PER_UPDATE = 1  # Количество эпох за одно обновление сети
GRADIENT_ACCUMULATION_STEPS = 4  # Количество шагов накопления градиентов

# --- Параметры Replay Buffer Server ---
MIN_REPLAY_BUFFER_SIZE = 2048  # ~5-7 игр перед стартом (distributed mode)
MAX_REPLAY_BUFFER_SIZE = 100000

# --- Параметры Model Server ---
CHECKPOINT_INTERVAL = 500  # Сохранять чекпоинт каждые N обновлений модели
INFERENCE_BATCH_SIZE = 64 # Размер батча для Inference Server (чем больше, тем эффективнее GPU)
INFERENCE_TIMEOUT = 0.01  # Максимальное время ожидания (сек) перед отправкой неполного батча

# --- Параметры Self-Play Worker ---
GAME_INTERVAL = 0  # Без паузы между играми для максимальной скорости
NUM_GAMES = 100000  # Общее количество игр для обучения (sequential mode)
MEMORY_SIZE = 50000  # Размер replay memory (sequential mode)
MIN_SAMPLES_FOR_TRAINING = 4096  # Минимум позиций перед началом обучения (sequential)

# --- Общие параметры ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELF_PLAY_DEVICE = "cuda"

# --- Параметры логирования и вывода ---
HTML_UPDATE_EVERY_N_MOVES = 20 # Как часто обновлять HTML-файл с доской

# --- Параметры сохранения ---
CHECKPOINT_PATH = "rl_checkpoint.pth"  # Прогресс обучения
MODEL_PATH = "rl_chess_model.pth"      # Веса модели для игры
MODEL_SAVE_PATH = MODEL_PATH           # Алиас для совместимости
SAVE_CHECKPOINT_EVERY_N_STEPS = 10     # Чекпоинт каждые N шагов (distributed)
SAVE_EVERY_N_GAMES = 5                 # Чекпоинт каждые N игр (sequential)
BACKUP_DIR = "backups"

# --- Устройства ---
TRAINING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'