# -*- coding: utf-8 -*-
"""
GUI для наблюдения за игрой модели против Stockfish.
Использует Pygame для визуализации.

Использование:
    python gui_test_vs_stockfish.py --elo 1320 --mcts 400
    python gui_test_vs_stockfish.py --skill 0 --mcts 800
"""
import sys
import os
import argparse
from collections import deque

import pygame
import chess
import chess.engine
import torch
import time

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent
from rl_chess import config

# --- Константы ---
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 600
BOARD_SIZE = 560
MARGIN = 20
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_X = BOARD_SIZE + MARGIN * 2

# Цвета
WHITE_SQUARE = (238, 238, 210)
BLACK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (186, 202, 68)
LAST_MOVE_COLOR = (255, 255, 100, 128)
BG_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)


def load_model(model_path, device):
    """Загружает модель."""
    net = ChessNetwork().to(device)
    
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл {model_path} не найден!")
        sys.exit(1)
    
    state_dict = torch.load(model_path, map_location=device)
    
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net.eval()
    return net


def find_stockfish():
    """Ищет Stockfish."""
    paths = [
        "chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
        "chess_engines/stockfish.exe",
        "stockfish.exe",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def draw_board(screen, board, last_move, view_white=True):
    """Рисует доску и фигуры."""
    files_range = range(8) if view_white else range(7, -1, -1)
    ranks_range = range(7, -1, -1) if view_white else range(8)
    
    piece_font = pygame.font.SysFont("segoeuisymbol", 58)
    coord_font = pygame.font.SysFont("arial", 14)
    
    for r_idx, r in enumerate(ranks_range):
        for f_idx, f in enumerate(files_range):
            square = chess.square(f, r)
            x = MARGIN + f_idx * SQUARE_SIZE
            y = MARGIN + r_idx * SQUARE_SIZE
            
            # Цвет клетки
            color = WHITE_SQUARE if (r + f) % 2 == 0 else BLACK_SQUARE
            
            # Подсветка последнего хода
            if last_move and (square == last_move.from_square or square == last_move.to_square):
                color = HIGHLIGHT_COLOR
            
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            # Координаты
            if f_idx == 0:
                text = coord_font.render(str(r + 1), True, (100, 100, 100))
                screen.blit(text, (x + 2, y + 2))
            if r_idx == 7:
                text = coord_font.render(chess.FILE_NAMES[f], True, (100, 100, 100))
                screen.blit(text, (x + SQUARE_SIZE - 12, y + SQUARE_SIZE - 16))
            
            # Фигура
            piece = board.piece_at(square)
            if piece:
                symbol = piece.unicode_symbol()
                text_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                
                # Обводка для белых фигур
                if piece.color == chess.WHITE:
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]:
                        outline = piece_font.render(symbol, True, (30, 30, 30))
                        rect = outline.get_rect(center=(x + SQUARE_SIZE//2 + dx, y + SQUARE_SIZE//2 + dy))
                        screen.blit(outline, rect)
                
                text = piece_font.render(symbol, True, text_color)
                rect = text.get_rect(center=(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2))
                screen.blit(text, rect)


def draw_info_panel(screen, game_info):
    """Рисует информационную панель справа."""
    font = pygame.font.SysFont("arial", 16)
    bold_font = pygame.font.SysFont("arial", 18, bold=True)
    
    x = INFO_PANEL_X
    y = MARGIN
    
    # Заголовок
    title = bold_font.render("AI vs Stockfish", True, TEXT_COLOR)
    screen.blit(title, (x, y))
    y += 35
    
    # Информация
    lines = [
        f"Ход: {game_info['move_count']}",
        f"",
        f"Белые: {game_info['white_player']}",
        f"Чёрные: {game_info['black_player']}",
        f"",
        f"Статус: {game_info['status']}",
        f"",
        f"Последний ход:",
        f"  {game_info['last_move']}",
    ]
    
    if game_info['result']:
        lines.append(f"")
        lines.append(f"РЕЗУЛЬТАТ: {game_info['result']}")
    
    for line in lines:
        text = font.render(line, True, TEXT_COLOR)
        screen.blit(text, (x, y))
        y += 22
    
    # Инструкции внизу
    y = SCREEN_HEIGHT - 100
    instructions = [
        "Пробел - пауза/продолжить",
        "R - новая игра",
        "F - перевернуть доску",
        "ESC - выход"
    ]
    for line in instructions:
        text = font.render(line, True, (150, 150, 150))
        screen.blit(text, (x, y))
        y += 20



def main():
    parser = argparse.ArgumentParser(description="GUI: Модель vs Stockfish")
    parser.add_argument("--model", type=str, default="rl_chess_model.pth")
    parser.add_argument("--elo", type=int, default=1320, help="ELO Stockfish (мин 1320)")
    parser.add_argument("--skill", type=int, default=None, help="Skill Level 0-20 (вместо ELO)")
    parser.add_argument("--mcts", type=int, default=400, help="MCTS симуляций")
    parser.add_argument("--delay", type=float, default=0.5, help="Задержка между ходами (сек)")
    parser.add_argument("--our-color", type=str, default="white", choices=["white", "black"],
                        help="Цвет нашей модели")
    args = parser.parse_args()
    
    # Инициализация Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RL Chess vs Stockfish")
    clock = pygame.time.Clock()
    
    # Загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print("Загрузка модели...")
    net = load_model(args.model, device)
    agent = MCTSAgent(net, device=device, num_simulations=args.mcts)
    print("Модель загружена!")
    
    # Загрузка Stockfish
    stockfish_path = find_stockfish()
    if not stockfish_path:
        print("ОШИБКА: Stockfish не найден!")
        sys.exit(1)
    
    print(f"Stockfish: {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Настройка силы
    if args.skill is not None:
        engine.configure({"Skill Level": args.skill})
        sf_strength = f"Skill {args.skill}"
    else:
        elo = max(1320, args.elo)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        sf_strength = f"ELO {elo}"
    
    print(f"Stockfish: {sf_strength}")
    
    # Определяем цвета
    our_color = chess.WHITE if args.our_color == "white" else chess.BLACK
    
    # Игровое состояние
    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    last_move = None
    move_count = 0
    game_over = False
    paused = False
    view_white = True
    result = None
    
    # Информация для панели
    game_info = {
        'move_count': 0,
        'white_player': "Наша модель" if our_color == chess.WHITE else f"Stockfish ({sf_strength})",
        'black_player': "Наша модель" if our_color == chess.BLACK else f"Stockfish ({sf_strength})",
        'status': "Игра идёт",
        'last_move': "-",
        'result': None
    }
    
    last_move_time = time.time()
    
    # Главный цикл
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    game_info['status'] = "Пауза" if paused else "Игра идёт"
                elif event.key == pygame.K_f:
                    view_white = not view_white
                elif event.key == pygame.K_r:
                    # Новая игра
                    board = chess.Board()
                    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
                    last_move = None
                    move_count = 0
                    game_over = False
                    paused = False
                    result = None
                    game_info['move_count'] = 0
                    game_info['status'] = "Игра идёт"
                    game_info['last_move'] = "-"
                    game_info['result'] = None
        
        # Логика игры
        if not game_over and not paused and time.time() - last_move_time > args.delay:
            move_count += 1
            
            if board.turn == our_color:
                # Ход нашей модели
                game_info['status'] = "Модель думает..."
                pygame.display.set_caption("RL Chess vs Stockfish - Модель думает...")
                
                # Отрисовка перед долгим вычислением
                screen.fill(BG_COLOR)
                draw_board(screen, board, last_move, view_white)
                draw_info_panel(screen, game_info)
                pygame.display.flip()
                
                move, _ = agent.get_move(board, board_history=history, temperature=0.1, is_self_play=False)
                player = "Модель"
            else:
                # Ход Stockfish
                game_info['status'] = "Stockfish думает..."
                result_sf = engine.play(board, chess.engine.Limit(time=0.1))
                move = result_sf.move
                player = "Stockfish"
            
            if move:
                board.push(move)
                history.append(board.copy())
                last_move = move
                game_info['move_count'] = move_count
                game_info['last_move'] = f"{player}: {move.uci()}"
                game_info['status'] = "Игра идёт"
                pygame.display.set_caption("RL Chess vs Stockfish")
            
            last_move_time = time.time()
            
            # Проверка конца игры
            if board.is_game_over(claim_draw=True):
                game_over = True
                result = board.result(claim_draw=True)
                game_info['result'] = result
                game_info['status'] = "Игра окончена"
                
                # Определяем победителя
                if result == "1-0":
                    winner = "Белые" if our_color == chess.WHITE else "Stockfish"
                elif result == "0-1":
                    winner = "Чёрные" if our_color == chess.BLACK else "Stockfish"
                else:
                    winner = "Ничья"
                
                print(f"\nИгра окончена! Результат: {result} ({winner})")
                print(f"Всего ходов: {move_count}")
        
        # Отрисовка
        screen.fill(BG_COLOR)
        draw_board(screen, board, last_move, view_white)
        draw_info_panel(screen, game_info)
        pygame.display.flip()
        
        clock.tick(60)
    
    # Завершение
    engine.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
