# -*- coding: utf-8 -*-
import torch
import chess
import argparse
from collections import deque

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent
from rl_chess import config

def play(model_path, simulations_per_move):
    """
    Главная функция для игры человека против ИИ.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Загрузка модели
    print("Загрузка обученной модели...")
    net = ChessNetwork().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Исправляем ключи если модель была сохранена с torch.compile() (префиксы "_orig_mod.")
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Обнаружена модель с torch.compile(), исправляем ключи...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Убираем префикс "_orig_mod."
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        net.load_state_dict(state_dict)
        print("Модель успешно загружена.")
    except FileNotFoundError:
        print(f"Ошибка: Файл модели не найден по пути: {model_path}")
        print("Пожалуйста, убедитесь, что вы сначала запустили RL_train.py для обучения и сохранения модели.")
        return
        
    agent = MCTSAgent(net, device=device, num_simulations=simulations_per_move)

    # 2. Выбор цвета
    player_color = None
    while player_color not in [chess.WHITE, chess.BLACK]:
        choice = input("Выберите цвет (w - белые, b - черные): ").lower()
        if choice == 'w':
            player_color = chess.WHITE
        elif choice == 'b':
            player_color = chess.BLACK

    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    
    # 3. Игровой цикл
    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        
        if board.turn == player_color:
            # Ход человека
            move = None
            while move is None:
                try:
                    uci_move = input("Ваш ход (в формате UCI, например, e2e4): ")
                    move = chess.Move.from_uci(uci_move)
                    if move not in board.legal_moves:
                        print("Нелегальный ход! Попробуйте еще раз.")
                        move = None
                except Exception as e:
                    print(f"Ошибка ввода: {e}. Пожалуйста, используйте формат UCI.")
                    move = None
            board.push(move)
            history.append(board.copy())
        else:
            # Ход ИИ
            print("ИИ думает...")
            ai_move, _ = agent.get_move(board, board_history=history)
            print(f"ИИ выбрал ход: {ai_move.uci()}")
            board.push(ai_move)
            history.append(board.copy())
            
    print("\n--- Игра окончена! ---")
    print(f"Результат: {board.result(claim_draw=True)}")
    print("Финальная позиция:")
    print(board)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Играть в шахматы против RL ИИ.")
    parser.add_argument("--model", type=str, default="rl_chess_model.pth", 
                        help="Путь к файлу с весами модели.")
    parser.add_argument("--simulations", type=int, default=800, 
                        help="Количество симуляций MCTS на каждый ход ИИ (значение по умолчанию ещё раз увеличено).")
    args = parser.parse_args()

    play(args.model, args.simulations) 