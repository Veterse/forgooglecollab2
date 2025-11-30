# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp
import numpy as np
import logging
from . import config

logger = logging.getLogger(__name__)

class SharedReplayBuffer:
    """
    Буфер воспроизведения, использующий общую память (shared memory) torch.
    Это позволяет избежать сериализации и копирования данных между процессами.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        
        # Общие счетчики и блокировка для потокобезопасности
        self.lock = mp.Lock()
        self.size = mp.Value('i', 0)
        self.head = mp.Value('i', 0)

        # Создаем тензоры в общей памяти
        state_shape = (config.INPUT_CHANNELS, 8, 8)
        policy_size = 4672 # Выход сети

        logger.info(f"Выделение общей памяти для буфера на {capacity} элементов...")
        
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32).share_memory_()
        self.policies = torch.zeros((capacity, policy_size), dtype=torch.float32).share_memory_()
        self.values = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()
        
        logger.info("Общая память успешно выделена.")

    def add(self, game_data):
        """Добавляет данные одной завершенной игры в буфер."""
        states, policies, values = zip(*game_data)
        num_moves = len(states)

        with self.lock:
            start_idx = self.head.value
            
            # Копируем данные в общую память, обрабатывая зацикливание буфера
            indices = torch.arange(start_idx, start_idx + num_moves) % self.capacity
            
            self.states[indices] = torch.stack(states)
            self.policies[indices] = torch.stack(policies)
            self.values[indices] = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

            # Обновляем указатель и размер
            self.head.value = (start_idx + num_moves) % self.capacity
            self.size.value = min(self.capacity, self.size.value + num_moves)
            
    def sample(self, batch_size):
        """Сэмплирует батч данных из буфера."""
        with self.lock:
            indices = np.random.randint(0, self.size.value, size=batch_size)
            
            # Данные копируются при перемещении на GPU, здесь лишнее копирование не нужно
            return (
                self.states[indices],
                self.policies[indices],
                self.values[indices].squeeze(-1) # Убираем лишнее измерение
            )

    def is_ready(self):
        """Проверяет, достаточно ли данных для начала обучения."""
        return self.size.value >= config.MIN_REPLAY_BUFFER_SIZE 