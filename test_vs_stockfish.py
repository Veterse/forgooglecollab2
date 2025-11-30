# -*- coding: utf-8 -*-
"""
Скрипт для тестирования модели против Stockfish.
Два режима:
1. Турнир (--games N) — быстрый матч из N партий
2. Визуальный (--visual) — показывает доску ход за ходом

Использование:
    python test_vs_stockfish.py --games 10 --elo 400
    python test_vs_stockfish.py --visual --elo 250
"""
import argparse
import chess
import chess.engine
import torch
import os
import sys
import time
from collections import deque

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent
from rl_chess import config


def load_model(model_path, device):
    """Загружает модель из чекпоинта."""
    print(f"Загрузка модели из {model_path}...")
    
    net = ChessNetwork().to(device)
    
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл {model_path} не найден!")
        sys.exit(1)
    
    state_dict = torch.load(model_path, map_location=device)
    
    # Убираем префикс _orig_mod. если модель была скомпилирована
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net.eval()
    print("Модель загружена!")
    return net


def find_stockfish():
    """Ищет Stockfish в стандартных местах."""
    possible_paths = [
        "chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
        "chess_engines/stockfish.exe",
        "chess_engines/stockfish-windows-x86-64-avx2.exe",
        "chess_engines/stockfish-windows-x86-64.exe",
        "stockfish.exe",
        "stockfish",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def print_board(board, last_move=None, clear=True):
    """Красиво выводит доску в консоль."""
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    
    for rank in range(7, -1, -1):
        print(f"{rank + 1} │", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            if piece:
                symbol = piece.unicode_symbol()
            else:
                symbol = " "
            
            # Подсветка последнего хода
            if last_move and (square == last_move.from_square or square == last_move.to_square):
                print(f"*{symbol}*│", end="")
            else:
                print(f" {symbol} │", end="")
        
        print()
        if rank > 0:
            print("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
    
    print("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    print("    a   b   c   d   e   f   g   h\n")


def play_game(agent, engine, our_color, mcts_sims, visual=False, move_time=0.5):
    """Играет одну партию. Возвращает результат с точки зрения нашей модели."""
    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    move_count = 0
    last_move = None
    
    while not board.is_game_over(claim_draw=True):
        move_count += 1
        
        if visual:
            print_board(board, last_move)
            turn_str = "Белые" if board.turn == chess.WHITE else "Чёрные"
            player = "Наша модель" if board.turn == our_color else "Stockfish"
            print(f"Ход {move_count}: {turn_str} ({player})")
        
        if board.turn == our_color:
            # Наш ход
            move, _ = agent.get_move(board, board_history=history, temperature=0.1, is_self_play=False)
            if visual:
                print(f"Модель играет: {move.uci()}")
        else:
            # Ход Stockfish
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            if visual:
                print(f"Stockfish играет: {move.uci()}")
        
        board.push(move)
        history.append(board.copy())
        last_move = move
        
        if visual:
            time.sleep(move_time)
    
    # Результат
    result = board.result(claim_draw=True)
    
    if visual:
        print_board(board, last_move)
        print(f"\n{'='*40}")
        print(f"ИГРА ОКОНЧЕНА: {result}")
        print(f"Всего ходов: {move_count}")
        print(f"{'='*40}\n")
    
    # Возвращаем результат с точки зрения нашей модели
    if result == "1-0":
        return 1 if our_color == chess.WHITE else -1
    elif result == "0-1":
        return 1 if our_color == chess.BLACK else -1
    else:
        return 0


def run_tournament(agent, engine, num_games, mcts_sims):
    """Проводит турнир из num_games партий."""
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\n{'='*50}")
    print(f"ТУРНИР: {num_games} партий против Stockfish")
    print(f"MCTS симуляций: {mcts_sims}")
    print(f"{'='*50}\n")
    
    for i in range(num_games):
        # Чередуем цвета
        our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        color_str = "белыми" if our_color == chess.WHITE else "чёрными"
        
        print(f"Партия {i+1}/{num_games} ({color_str})...", end=" ", flush=True)
        
        result = play_game(agent, engine, our_color, mcts_sims, visual=False)
        
        if result == 1:
            wins += 1
            print("ПОБЕДА!")
        elif result == -1:
            losses += 1
            print("Поражение")
        else:
            draws += 1
            print("Ничья")
    
    # Итоги
    print(f"\n{'='*50}")
    print(f"ИТОГИ ТУРНИРА")
    print(f"{'='*50}")
    print(f"Победы:    {wins}")
    print(f"Поражения: {losses}")
    print(f"Ничьи:     {draws}")
    print(f"Winrate:   {wins/num_games*100:.1f}%")
    print(f"{'='*50}\n")
    
    return wins, losses, draws


def main():
    # Очистка консоли в начале
    os.system('cls' if os.name == 'nt' else 'clear')
    
    parser = argparse.ArgumentParser(description="Тест модели против Stockfish")
    parser.add_argument("--model", type=str, default="rl_chess_model.pth",
                        help="Путь к модели")
    parser.add_argument("--stockfish", type=str, default=None,
                        help="Путь к Stockfish (автопоиск если не указан)")
    parser.add_argument("--elo", type=int, default=400,
                        help="ELO Stockfish (250-3000)")
    parser.add_argument("--games", type=int, default=10,
                        help="Количество партий в турнире")
    parser.add_argument("--mcts", type=int, default=400,
                        help="Количество MCTS симуляций")
    parser.add_argument("--visual", action="store_true",
                        help="Визуальный режим (одна партия с отображением)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Задержка между ходами в визуальном режиме (сек)")
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    # Загрузка модели
    net = load_model(args.model, device)
    agent = MCTSAgent(net, device=device, num_simulations=args.mcts)
    
    # Поиск Stockfish
    stockfish_path = args.stockfish or find_stockfish()
    if not stockfish_path:
        print("\nОШИБКА: Stockfish не найден!")
        print("Скачай с https://stockfishchess.org/download/")
        print("и положи в папку chess_engines/")
        sys.exit(1)
    
    print(f"Stockfish: {stockfish_path}")
    
    # Запуск движка
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Настройка силы Stockfish
    # Stockfish 16+ минимум 1320 ELO, используем Skill Level для слабой игры
    # Skill Level 0-20, где 0 = самый слабый (~1350 ELO), 20 = полная сила
    if args.elo < 1320:
        # Конвертируем желаемый ELO в Skill Level (примерно)
        # 250 ELO -> Skill 0, 1320 ELO -> Skill 5
        skill = max(0, min(20, (args.elo - 250) // 200))
        engine.configure({"Skill Level": skill})
        print(f"Stockfish Skill Level: {skill} (запрошено ~{args.elo} ELO)")
        print("Примечание: Stockfish 16+ минимум 1320 ELO, используем Skill Level")
    else:
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": args.elo
        })
        print(f"Stockfish ELO: {args.elo}")
    
    try:
        if args.visual:
            # Визуальный режим — одна партия
            print("\nВизуальный режим. Нажми Enter для старта...")
            input()
            play_game(agent, engine, chess.WHITE, args.mcts, visual=True, move_time=args.delay)
        else:
            # Турнир
            run_tournament(agent, engine, args.games, args.mcts)
    
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
