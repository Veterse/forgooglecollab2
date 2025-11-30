# -*- coding: utf-8 -*-
"""
Процесс-тренер (Training Worker).

Этот процесс отвечает за непрерывное обучение нейронной сети.
Он запрашивает батчи данных у ReplayBufferServer, выполняет шаги
оптимизации на GPU и периодически отправляет обновленные веса
модели в ModelServer.
"""
import multiprocessing
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import time
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss, CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

import rl_chess.config as config
from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_utils import setup_worker_logging
from rl_chess.shared_buffer import SharedReplayBuffer

class TrainingWorker(multiprocessing.Process):
    """
    Процесс, отвечающий за обучение общей модели.
    """
    def __init__(self, model: ChessNetwork, replay_buffer: SharedReplayBuffer, optimizer, scheduler, training_step_counter: multiprocessing.Value, total_games_played_counter: multiprocessing.Value, stats_counters=None):
        super().__init__()
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_step_counter = training_step_counter
        self.total_games_played_counter = total_games_played_counter
        self.name = "TrainingWorker"
        self.scaler = None # Для смешанной точности
        self.white_wins = None
        self.black_wins = None
        self.draws = None
        if stats_counters is not None:
            self.white_wins, self.black_wins, self.draws = stats_counters

    def run(self):
        """
        Основной цикл жизни процесса.
        """
        setup_worker_logging()
        device = torch.device(config.TRAINING_DEVICE)
        self.model.to(device)

        # <<< НАЧАЛО ИЗМЕНЕНИЙ
        # ОТКЛЮЧАЕМ Mixed Precision, так как она ломает обучение Policy Head
        # use_bfloat16 = (device.type == 'cuda' and torch.cuda.is_bf16_supported())
        use_bfloat16 = False
        
        # GradScaler нужен для float16, но безопасен и для bfloat16 (хотя и менее критичен)
        # self.scaler = GradScaler(enabled=use_bfloat16)
        self.scaler = None # Отключаем scaler
        
        log_message = f"🚀 Процесс запущен на [{device}]. "
        log_message += "Смешанная точность: Выключена (используется float32) [FORCED FIX]"
        logging.info(log_message)
        # <<< КОНЕЦ ИЗМЕНЕНИЙ

        # Отслеживаем сколько данных было при последнем обучении
        last_trained_buffer_size = 0
        training_start_time = time.time()
        last_log_time = time.time()
        
        while True:
            if not self.replay_buffer.is_ready():
                logging.info(f"Ожидание наполнения буфера... ({self.replay_buffer.size.value}/{config.MIN_REPLAY_BUFFER_SIZE})")
                time.sleep(5)
                continue
            
            # RATE LIMITING: ждём пока накопится достаточно НОВЫХ данных
            # Это предотвращает переобучение на одних и тех же позициях
            current_buffer_size = self.replay_buffer.size.value
            new_samples = current_buffer_size - last_trained_buffer_size
            
            # Требуем минимум TRAIN_BATCH_SIZE новых позиций перед следующим шагом
            if new_samples < config.TRAIN_BATCH_SIZE:
                time.sleep(0.5)  # Короткая пауза, чтобы не спамить CPU
                continue
            
            current_step = self.training_step_counter.value
            games_played = self.total_games_played_counter.value
            
            # Логируем статистику каждые 30 секунд
            if time.time() - last_log_time > 30:
                elapsed = time.time() - training_start_time
                games_per_hour = (games_played / elapsed) * 3600 if elapsed > 0 else 0
                steps_per_hour = (current_step / elapsed) * 3600 if elapsed > 0 else 0
                logging.info(f"📊 СТАТИСТИКА: Игр: {games_played} | Шагов: {current_step} | "
                           f"Буфер: {current_buffer_size} | "
                           f"Скорость: {games_per_hour:.1f} игр/час, {steps_per_hour:.1f} шагов/час")
                last_log_time = time.time()
            
            batch = self.replay_buffer.sample(config.TRAIN_BATCH_SIZE)

            # <<< ИЗМЕНЕНИЕ ЗДЕСЬ
            # Передаем флаг use_bfloat16 в метод update_network
            self.update_network(batch, device, use_bfloat16)
            # <<< КОНЕЦ ИЗМЕНЕНИЯ
            
            # Обновляем счётчик после успешного обучения
            last_trained_buffer_size = current_buffer_size
            
            if self.training_step_counter.value % config.SAVE_CHECKPOINT_EVERY_N_STEPS == 0:
                self._save_checkpoint()

    def _save_checkpoint(self):
        """
        Атомарно сохраняет состояние модели, оптимизатора и счетчиков для возобновления.
        """
        try:
            logging.info(f"💾 Сохранение чекпоинта на шаге {self.training_step_counter.value}...")
            
            # Данные для сохранения
            checkpoint_data = {
                'training_steps': self.training_step_counter.value,
                'games_played': self.total_games_played_counter.value,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }

            if self.white_wins is not None and self.black_wins is not None and self.draws is not None:
                checkpoint_data.update({
                    'white_wins': self.white_wins.value,
                    'black_wins': self.black_wins.value,
                    'draws': self.draws.value,
                })

            # Атомарное сохранение: сначала во временный файл, потом переименовываем
            temp_checkpoint_path = config.CHECKPOINT_PATH + ".tmp"
            
            torch.save(checkpoint_data, temp_checkpoint_path)
            # Также сохраняем чистую модель для инференса
            torch.save(self.model.state_dict(), config.MODEL_PATH)
            
            os.replace(temp_checkpoint_path, config.CHECKPOINT_PATH)
            logging.info(f"✅ Чекпоинт и модель успешно сохранены в '{config.CHECKPOINT_PATH}' и '{config.MODEL_PATH}'.")

        except Exception as e:
            logging.error(f"❌ Ошибка при сохранении чекпоинта: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)

    def update_network(self, batch, device, use_bfloat16):
        """
        Выполняет один шаг обновления весов общей модели, блокируя ее на время обновления.
        """
        self.model.train()
        
        states, policy_targets, value_targets = batch
        
        states = states.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        self.optimizer.zero_grad()

        # <<< ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ
        # Явно указываем dtype=torch.bfloat16 для autocast
        # with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        # <<< КОНЕЦ ИЗМЕНЕНИЯ
        # Сеть возвращает log_softmax для политики и tanh для value
        log_policy_preds, value_preds = self.model(states)
            
        # 1. Потери значения (Value Loss) - Mean Squared Error
        value_loss = MSELoss()(value_preds.squeeze(), value_targets)
            
        # 2. Потери политики (Policy Loss) - кросс-энтропия с распределением MCTS
        policy_loss = -torch.sum(policy_targets * log_policy_preds) / states.size(0)

        loss = value_loss + policy_loss

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        
        loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        
        with self.training_step_counter.get_lock():
            self.training_step_counter.value += 1
        
        current_step = self.training_step_counter.value
        if current_step % 50 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"💡 Шаг {current_step} | Loss: {loss.item():.4f} (v:{value_loss.item():.4f} p:{policy_loss.item():.4f}) | LR: {current_lr:.2e}")
        
        return loss.item()