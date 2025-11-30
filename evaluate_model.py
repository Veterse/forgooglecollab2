
# -*- coding: utf-8 -*-
import torch
import chess
import logging
import sys
import os
from datetime import datetime
from collections import deque
from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent
import rl_chess.config as config

def evaluate():
    print("--- ЗАПУСК ДЕТЕРМИНИРОВАННОЙ ОЦЕНКИ МОДЕЛИ ---")
    
    # 1. Настройка устройства и модели
    device = torch.device(config.TRAINING_DEVICE)
    print(f"Используется устройство: {device}")
    
    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель {model_path} не найдена!")
        return

    model = ChessNetwork().to(device)
    try:
        # Загружаем веса (с учетом возможного префикса _orig_mod от компиляции)
        state_dict = torch.load(model_path, map_location=device)
        # Очистка ключей от префикса _orig_mod. (если вдруг сохранили скомпилированную)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return

    model.eval()
    
    # 2. Создаем агента (без шума, strict mode)
    # Важно: is_self_play=False в методе get_move отключит шум Дирихле
    agent = MCTSAgent(model, device=device, num_simulations=config.MCTS_SIMULATIONS)
    
    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    moves_log = []
    
    print("Начало партии...")
    
    while not board.is_game_over(claim_draw=True):
        # Temperature = 0.0 заставляет выбирать строго самый посещаемый ход
        # is_self_play = False отключает шум Дирихле
        move, _ = agent.get_move(board, board_history=history, temperature=0.0, is_self_play=False)
        
        if move is None:
            print("Агент сдался (не нашел хода).")
            break
            
        board.push(move)
        history.append(board.copy())
        moves_log.append(move.uci())
        
        if len(moves_log) % 10 == 0:
            print(f"Ход {len(moves_log)}: {move.uci()}")

    result = board.result(claim_draw=True)
    print(f"Партия завершена. Результат: {result}")
    print(f"Всего ходов: {len(moves_log)}")
    
    # 3. Запись в лог
    log_file = "evaluation_history.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
==================================================
Дата: {timestamp}
Результат: {result}
Длина: {len(moves_log)} ходов
Последний FEN: {board.fen()}
Ходы: {" ".join(moves_log)}
==================================================
"""
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)
        
    print(f"Результат записан в {log_file}")

if __name__ == "__main__":
    # Отключаем лишний шум от библиотек
    logging.basicConfig(level=logging.ERROR)
    evaluate()
