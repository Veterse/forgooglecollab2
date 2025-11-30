# -*- coding: utf-8 -*-
import sys
import os
import logging
from collections import deque

# Добавляем родительскую директорию в sys.path, чтобы разрешить импорты
# Это позволяет запускать скрипт напрямую, как python rl_chess/RL_gui_play.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import chess
import argparse
import torch
import time
import math

from rl_chess.RL_network import ChessNetwork
from rl_chess.RL_agent import MCTSAgent
from rl_chess import config

thinking_logger = logging.getLogger("gui_thinking")
if thinking_logger.hasHandlers():
    thinking_logger.handlers.clear()
fh = logging.FileHandler("gui_mcts_log.txt", mode="a", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(message)s"))
thinking_logger.addHandler(fh)
thinking_logger.setLevel(logging.INFO)

# --- Константы ---
SCREEN_SIZE = 600
BOARD_SIZE = 560
MARGIN = (SCREEN_SIZE - BOARD_SIZE) // 2
SQUARE_SIZE = BOARD_SIZE // 8
ARROW_COLOR = (255, 102, 102, 200) # Полупрозрачный красный

# --- Настройка Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("RL Chess AI")

def draw_arrows(surface, arrows):
    """ Рисует стрелки на доске. """
    for start_pos, end_pos in arrows:
        # Рисуем линию
        pygame.draw.line(surface, ARROW_COLOR, start_pos, end_pos, 7)
        
        # Рисуем наконечник стрелки
        angle = math.atan2(start_pos[1] - end_pos[1], start_pos[0] - end_pos[0])
        # Точки для треугольника наконечника
        p1 = (end_pos[0] + 20 * math.cos(angle - math.pi / 6), end_pos[1] + 20 * math.sin(angle - math.pi / 6))
        p2 = (end_pos[0] + 20 * math.cos(angle + math.pi / 6), end_pos[1] + 20 * math.sin(angle + math.pi / 6))
        pygame.draw.polygon(surface, ARROW_COLOR, (end_pos, p1, p2))

def draw_promotion_choice(surface, square, player_color, view_is_white):
    """ Рисует меню выбора фигуры для превращения пешки. """
    file = chess.square_file(square)

    # Определяем позицию меню в экранных координатах, учитывая поворот доски
    if view_is_white:
        # Доска не перевернута. Меню появляется вверху доски.
        x = MARGIN + file * SQUARE_SIZE
        y = MARGIN
    else: # Доска перевернута
        # Меню появляется внизу доски и рисуется вверх.
        x = MARGIN + (7 - file) * SQUARE_SIZE
        y = MARGIN + (8 - 4) * SQUARE_SIZE # Верхняя точка меню на 4-й клетке от верха.

    # Фон для меню
    menu_rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE * 4)
    pygame.draw.rect(surface, (200, 200, 200), menu_rect)
    pygame.draw.rect(surface, (50, 50, 50), menu_rect, 2)

    # Рисуем фигуры
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    piece_symbols = [chess.Piece(p, player_color).unicode_symbol() for p in promotion_pieces]
    
    piece_font = pygame.font.SysFont("segoeuisymbol", 50)
    text_color = (255, 255, 255) if player_color == chess.WHITE else (0, 0, 0)

    for i, symbol in enumerate(piece_symbols):
        piece_y = y + i * SQUARE_SIZE + SQUARE_SIZE // 2
        text_surface = piece_font.render(symbol, True, text_color)
        text_rect = text_surface.get_rect(center=(x + SQUARE_SIZE // 2, piece_y))
        surface.blit(text_surface, text_rect)
        
    return menu_rect, promotion_pieces

def draw_board(board, selected_square, view_is_white):
    """ Отрисовывает доску, фигуры и легальные ходы в правильном порядке. """
    board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
    
    # Вид доски зависит от view_is_white, а не от цвета игрока
    files_range = range(8) if view_is_white else range(7, -1, -1)
    ranks_range = range(7, -1, -1) if view_is_white else range(8)

    # 1. Сначала рисуем все клетки
    for r_idx, r in enumerate(ranks_range):
        for f_idx, f in enumerate(files_range):
            square = chess.square(f, r)
            color = (238, 238, 210) if (r + f) % 2 == 0 else (118, 150, 86)
            
            # Подсветка выбранной клетки
            if selected_square is not None and square == selected_square:
                color = (186, 202, 68) # Желтый для выбранной

            pygame.draw.rect(board_surface, color, (f_idx * SQUARE_SIZE, r_idx * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # 2. Затем рисуем подсветку легальных ходов (поверх клеток)
    if selected_square is not None:
        # Получаем легальные ходы для выбранной фигуры
        legal_moves = [move for move in board.legal_moves if move.from_square == selected_square]
        for move in legal_moves:
            # Конвертируем конечную клетку хода в экранные координаты
            to_square = move.to_square
            to_file_orig = chess.square_file(to_square)
            to_rank_orig = chess.square_rank(to_square)
            
            # Инвертируем координаты в зависимости от вида
            f_idx_to = to_file_orig if view_is_white else 7 - to_file_orig
            r_idx_to = 7 - to_rank_orig if view_is_white else to_rank_orig

            # Рисуем полупрозрачный кружок
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 0, 0, 80), (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 5)
            board_surface.blit(s, (f_idx_to * SQUARE_SIZE, r_idx_to * SQUARE_SIZE))

    # 3. В последнюю очередь рисуем фигуры (поверх всего)
    for r_idx, r in enumerate(ranks_range):
        for f_idx, f in enumerate(files_range):
            square = chess.square(f, r)
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.unicode_symbol()
                text_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                piece_font = pygame.font.SysFont("segoeuisymbol", 64)
                
                center_pos = (f_idx * SQUARE_SIZE + SQUARE_SIZE // 2, r_idx * SQUARE_SIZE + SQUARE_SIZE // 2)
                
                # Добавляем обводку для белых фигур для лучшей читаемости
                if piece.color == chess.WHITE:
                    outline_color = (30, 30, 30)
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                        outline_surface = piece_font.render(piece_symbol, True, outline_color)
                        outline_rect = outline_surface.get_rect(center=(center_pos[0] + dx, center_pos[1] + dy))
                        board_surface.blit(outline_surface, outline_rect)

                text_surface = piece_font.render(piece_symbol, True, text_color)
                text_rect = text_surface.get_rect(center=center_pos)
                board_surface.blit(text_surface, text_rect)
                
    return board_surface

def square_to_center_pos(square, view_is_white):
    """ Конвертирует индекс клетки в центральные экранные координаты, учитывая вид. """
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    if view_is_white:
        screen_file_idx = file
        screen_rank_idx = 7 - rank
    else: # Вид за черных, доска перевернута
        screen_file_idx = 7 - file
        screen_rank_idx = rank
        
    x = MARGIN + screen_file_idx * SQUARE_SIZE + SQUARE_SIZE // 2
    y = MARGIN + screen_rank_idx * SQUARE_SIZE + SQUARE_SIZE // 2
    return (x, y)

def get_square_from_mouse(pos, view_is_white):
    """ Конвертирует координаты мыши в индекс клетки на доске. """
    x, y = pos
    if not (MARGIN < x < SCREEN_SIZE - MARGIN and MARGIN < y < SCREEN_SIZE - MARGIN):
        return None
    
    file_idx = (x - MARGIN) // SQUARE_SIZE
    rank_idx = (y - MARGIN) // SQUARE_SIZE
    
    if view_is_white:
        return chess.square(file_idx, 7 - rank_idx)
    else:
        return chess.square(7 - file_idx, rank_idx)

def main(model_path, simulations, player_color_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Загрузка ИИ ---
    print("Загрузка RL модели...")
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
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден: '{model_path}'")
        sys.exit(1)
    ai = MCTSAgent(net, device=device, num_simulations=simulations, thinking_logger=thinking_logger, log_mcts=True)

    # --- Выбор цвета игрока ---
    if player_color_choice is None:
        print("\nВЫБОР ЦВЕТА ДЛЯ GUI ИГРЫ:")
        player_color = None
        while player_color not in [chess.WHITE, chess.BLACK]:
            choice = input("Выберите ваш цвет (w - белые, b - черные): ").lower()
            if choice == 'w':
                player_color = chess.WHITE
                print("Вы играете за белых! GUI окно откроется через 2 секунды...")
            elif choice == 'b':
                player_color = chess.BLACK
                print("Вы играете за черных! GUI окно откроется через 2 секунды...")
            else:
                print("Введите 'w' для белых или 'b' для черных")
    else:
        player_color = chess.WHITE if player_color_choice == 'w' else chess.BLACK
        print(f"Цвет задан через командную строку: {'белые' if player_color == chess.WHITE else 'черные'}")
    
    # Небольшая пауза если цвет выбран в консоли
    if player_color_choice is None:
        time.sleep(2)

    # --- Игровое состояние ---
    board = chess.Board()
    selected_square = None
    arrows = []
    arrow_start_pos = None
    promotion_move = None # Хранит ход, ожидающий выбора фигуры
    view_is_white = True # Начинаем с вида за белых. 'F' для переворота.
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    
    # Обновляем заголовок окна с выбранным цветом
    color_text = "белые" if player_color == chess.WHITE else "черные"
    initial_turn_text = "Ваш ход" if player_color == chess.WHITE else "Ход ИИ"
    pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - {initial_turn_text}")

    # --- Игровой цикл ---
    running = True
    while running:
        is_player_turn = (board.turn == player_color)
        
        # Поверхность для рисования стрелок (чтобы они были поверх доски)
        arrow_surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
        if arrow_start_pos and pygame.mouse.get_pressed()[2]: # Если правая кнопка зажата
            draw_arrows(arrow_surface, [(arrow_start_pos, pygame.mouse.get_pos())])
        draw_arrows(arrow_surface, arrows)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    view_is_white = not view_is_white # Переворачиваем доску

            # --- Логика превращения пешки (имеет приоритет) ---
            if promotion_move:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    choice_rect, pieces = draw_promotion_choice(screen, promotion_move.to_square, player_color, view_is_white)
                    if choice_rect.collidepoint(event.pos):
                        # Определяем, на какую фигуру кликнули
                        rel_y = event.pos[1] - choice_rect.top
                        choice_idx = rel_y // SQUARE_SIZE
                        promotion_move.promotion = pieces[choice_idx]
                        board.push(promotion_move)
                        history.append(board.copy())
                        promotion_move = None # Сбрасываем состояние
                continue # Пропускаем остальную обработку

            # --- Логика рисования стрелок ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                arrow_start_pos = event.pos
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if arrow_start_pos:
                    # Конвертируем в координаты доски для чистоты
                    start_sq = get_square_from_mouse(arrow_start_pos, view_is_white)
                    end_sq = get_square_from_mouse(event.pos, view_is_white)
                    if start_sq is not None and end_sq is not None and start_sq != end_sq:
                         # Переводим в центр клеток, УЧИТЫВАЯ ПОВОРОТ ДОСКИ
                        start_center = square_to_center_pos(start_sq, view_is_white)
                        end_center = square_to_center_pos(end_sq, view_is_white)
                        arrows.append((start_center, end_center))
                    arrow_start_pos = None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Очищаем стрелки при любом левом клике
                arrows.clear()
                if is_player_turn and not board.is_game_over():
                    clicked_square = get_square_from_mouse(event.pos, view_is_white)
                    if clicked_square is not None:
                        if selected_square is not None:
                            move = chess.Move(selected_square, clicked_square)
                            
                            is_promotion = (board.piece_at(selected_square).piece_type == chess.PAWN and
                                            (chess.square_rank(clicked_square) == 7 or chess.square_rank(clicked_square) == 0))
                            
                            if is_promotion and chess.Move(move.from_square, move.to_square, chess.QUEEN) in board.legal_moves:
                                promotion_move = move # Сохраняем ход и ждем выбора
                            elif move in board.legal_moves:
                                board.push(move)
                                history.append(board.copy())
                            
                            selected_square = None # Снимаем выделение в любом случае

                        # Если клик не является легальным ходом, проверяем, не кликнули ли по другой своей фигуре
                        elif board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                            selected_square = clicked_square
                        else:
                            # Если клик по пустой клетке или по фигуре противника - снимаем выделение
                            selected_square = None

        # --- Ход ИИ ---
        if not is_player_turn and not board.is_game_over() and not promotion_move:
            pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - ИИ думает...")
            ai_move, _ = ai.get_move(board, board_history=history)
            if ai_move:
                board.push(ai_move)
                history.append(board.copy())
            pygame.display.set_caption(f"RL Chess AI - Вы играете за {color_text} - Ваш ход")

        # --- Отрисовка ---
        screen.fill((40, 40, 40))
        board_surface = draw_board(board, selected_square, view_is_white)
        screen.blit(board_surface, (MARGIN, MARGIN))
        screen.blit(arrow_surface, (0, 0)) # Рисуем стрелки поверх доски
        
        if promotion_move:
            draw_promotion_choice(screen, promotion_move.to_square, player_color, view_is_white)

        if board.is_game_over(claim_draw=True):
            font = pygame.font.SysFont(None, 60)
            result = board.result(claim_draw=True)
            outcome = ""
            if board.is_checkmate():
                outcome = "Мат!"
            elif board.is_stalemate():
                outcome = "Пат!"
            elif board.is_insufficient_material():
                outcome = "Ничья (недостаточно фигур)"
            elif board.is_seventyfive_moves():
                outcome = "Ничья (правило 75 ходов)"
            elif board.is_fivefold_repetition():
                outcome = "Ничья (пятикратное повторение)"
            
            text_surface = font.render(f"Игра окончена: {outcome} ({result})", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(SCREEN_SIZE/2, SCREEN_SIZE/2))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Играть против RL ИИ с графическим интерфейсом.")
    parser.add_argument("--model", type=str, default="rl_chess_model.pth", help="Путь к файлу модели.")

    parser.add_argument("--simulations", type=int, default=900, help="Количество симуляций MCTS за ход.")
    parser.add_argument("--color", type=str, choices=['w', 'b'], default=None, help="Ваш цвет (w - белые, b - черные). Если не указан, будет предложен выбор.")

    args = parser.parse_args()
    
    main(args.model, args.simulations, args.color) 