# -*- coding: utf-8 -*-
"""
Единый файл конфигурации для всех компонентов системы.
Поддержка TPU, CUDA и CPU.
"""

import torch

# --- Определение устройства (TPU/CUDA/CPU) ---
def get_device():
    """Определяет лучшее доступное устройство. Вызывать лениво, не при импорте!"""
    # Пробуем TPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return device, 'tpu'
    except (ImportError, RuntimeError):
        # RuntimeError если TPU занят или недоступен
        pass
    except Exception:
        pass
    
    # Пробуем CUDA
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    
    # Fallback на CPU
    return torch.device('cpu'), 'cpu'

# НЕ вызываем get_device() при импорте — это вызовет ошибку если TPU занят
# Каждый процесс должен вызвать get_device() сам когда будет готов
DEVICE = None  # Будет установлено при первом использовании
DEVICE_TYPE = None

# --- Представление состояния ---
BOARD_HISTORY_LENGTH = 8  # Сколько последних позиций подаем сети
PIECE_CHANNELS_PER_POSITION = 12  # 6 типов фигур на цвет
AUXILIARY_CHANNELS = 6  # Чей ход, halfmove clock, рокировки
INPUT_CHANNELS = BOARD_HISTORY_LENGTH * PIECE_CHANNELS_PER_POSITION + AUXILIARY_CHANNELS

# --- Параметры MCTS ---
MCTS_SIMULATIONS = 150  # Количество симуляций MCTS
MCTS_BATCH_SIZE = 16    # Батч для инференса в MCTS

# --- Параметры обучения ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 256
BATCH_SIZE = 256
SCHEDULER_GAMMA = 1.0
EPOCHS_PER_UPDATE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# --- Параметры Replay Buffer ---
MIN_REPLAY_BUFFER_SIZE = 2048
MAX_REPLAY_BUFFER_SIZE = 100000

# --- Параметры Model Server ---
CHECKPOINT_INTERVAL = 500
INFERENCE_BATCH_SIZE = 64
INFERENCE_TIMEOUT = 0.01

# --- Параметры Self-Play ---
GAME_INTERVAL = 0
NUM_GAMES = 100000
MEMORY_SIZE = 50000
MIN_SAMPLES_FOR_TRAINING = 4096

# --- Параметры логирования ---
HTML_UPDATE_EVERY_N_MOVES = 20

# --- Параметры сохранения ---
CHECKPOINT_PATH = "rl_checkpoint.pth"
MODEL_PATH = "rl_chess_model.pth"
MODEL_SAVE_PATH = MODEL_PATH
SAVE_CHECKPOINT_EVERY_N_STEPS = 10
SAVE_EVERY_N_GAMES = 5
BACKUP_DIR = "backups"

# --- Устройства (для совместимости со старым кодом) ---
# Эти значения устанавливаются динамически при запуске
TRAINING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SELF_PLAY_DEVICE = 'cpu'  # Self-play воркеры всегда на CPU