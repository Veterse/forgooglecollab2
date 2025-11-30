# -*- coding: utf-8 -*-
"""
Процесс-игрок (Self-Play Worker).

Этот процесс бесконечно играет партии сам с собой, используя последнюю
версию нейросети, и отправляет сгенерированные данные в ReplayBufferServer.
"""
import multiprocessing
import torch
import chess
import logging
import time
import os
from collections import deque

import rl_chess.config as config
from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent
from rl_chess.RL_utils import get_live_logger, format_move, setup_worker_logging
from rl_chess.shared_buffer import SharedReplayBuffer
from rl_chess.inference_server import PredictionClient

class SelfPlayWorker(multiprocessing.Process):
    """
    Процесс, отвечающий за self-play.
    Генерирует игровые данные, играя сам с собой, и отправляет их в ReplayBuffer.
    """
    def __init__(self, worker_id, input_queue, output_queue, replay_buffer: SharedReplayBuffer, total_games_played_counter, shared_inference_buffer, stats_counters):
        super().__init__()
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.replay_buffer = replay_buffer
        self.total_games_played_counter = total_games_played_counter
        self.shared_inference_buffer = shared_inference_buffer
        self.white_wins, self.black_wins, self.draws = stats_counters
        self.name = f"SelfPlayWorker-{worker_id}"
        self.logger = None

    def run(self):
        """
        Основной цикл рабочего процесса.
        """
        setup_worker_logging() # Настраиваем основной лог
        # Создаем логгер для live-апдейтов ВНУТРИ дочернего процесса
        self.logger = get_live_logger('live_updates.log', f"Player-{self.worker_id}")

        # Инициализируем клиент предсказаний
        predictor = PredictionClient(self.worker_id, self.input_queue, self.output_queue, self.shared_inference_buffer)
        
        # Устройство для воркера (CPU для логики, предсказания на GPU через сервер)
        device = torch.device("cpu")
        
        # Инициализируем агента без модели, но с predictor
        agent = MCTSAgent(model=None, device=device, num_simulations=config.MCTS_SIMULATIONS, predictor=predictor)
        
        logging.info(f"🚀 Процесс запущен. Предсказания через InferenceServer.")

        while True:
            # Логируем начало игры
            with self.total_games_played_counter.get_lock():
                self.total_games_played_counter.value += 1
                current_game_number = self.total_games_played_counter.value
            logging.info(f"--- Начало игры #{current_game_number} ---")

            # Процесс игры (Self-Play)
            board = chess.Board()
            history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
            game_memory = []
            move_count = 0
            
            while not board.is_game_over(claim_draw=True):
                move_count += 1
                
                # Устанавливаем температуру в зависимости от номера хода
                # Плавное затухание для лучшего исследования
                if move_count <= 15:
                    temperature = 1.2
                elif move_count <= 30:
                    temperature = 0.5
                else:
                    temperature = 0.1

                # Получаем ход через MCTS (запросы к сети пойдут через сервер)
                move, policy_target = agent.get_move(board, board_history=history, temperature=temperature, is_self_play=True)
                
                # Если агент не смог предложить ход, досрочно завершаем партию
                if move is None:
                    logging.warning("Агент не смог выбрать ход, игра прекращена.")
                    break
                
                # Логируем только каждый 50-й ход для отладки (в live_updates.log)
                if move_count % 50 == 0:
                    self.logger.info(f"Игра #{current_game_number} | Ход #{move_count}: {format_move(move)}")

                state_tensor = board_to_tensor(history, device)
                # Убеждаемся что policy на CPU для shared memory
                game_memory.append([state_tensor, policy_target.cpu()])
                board.push(move)
                history.append(board.copy())

            # Обработка результата и отправка данных
            total_games = self.total_games_played_counter.value
            
            result_str = board.result(claim_draw=True)
            value = 0
            if result_str == "1-0":
                value = 1
                with self.white_wins.get_lock(): self.white_wins.value += 1
            elif result_str == "0-1":
                value = -1
                with self.black_wins.get_lock(): self.black_wins.value += 1
            else:
                with self.draws.get_lock(): self.draws.value += 1

            logging.info(
                f"Игра #{total_games} завершена. "
                f"Результат: {result_str} ({move_count} ходов). "
                f"Общий счет: +{self.white_wins.value} -{self.black_wins.value} ={self.draws.value}"
            )
            
            # Каждый тензор состояния нужно отсоединить от графа вычислений и перенести на CPU
            final_game_memory = []
            for i, data in enumerate(game_memory):
                # Ценность меняется для каждого хода (1 для победителя, -1 для проигравшего)
                current_value = value if i % 2 == 0 else -value
                state_tensor, policy = data
                final_game_memory.append([state_tensor.detach().cpu(), policy, current_value])

            # Напрямую добавляем данные в общий буфер
            self.replay_buffer.add(final_game_memory)
            
            time.sleep(config.GAME_INTERVAL) 