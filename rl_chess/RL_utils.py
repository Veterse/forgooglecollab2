# -*- coding: utf-8 -*-
import chess
import logging

# Размер выхода политики, как в AlphaZero (73 типа ходов * 64 клетки)
POLICY_OUTPUT_SIZE = 4672

# Создаем карты для преобразования ходов в индексы и обратно
MOVE_TO_INDEX_MAP = {}
INDEX_TO_MOVE_MAP = {}

def _build_move_maps():
    """
    Создает детерминированные и быстрые карты для преобразования ходов.
    Логика основана на 73 "плоскостях" для каждого из 64 полей.
    """
    
    # 1. Ходы "королевы" (скольжение по прямым и диагоналям)
    # 8 направлений * 7 возможных расстояний = 56 плоскостей
    # N, NE, E, SE, S, SW, W, NW
    queen_directions = [8, 9, 1, -7, -8, -9, -1, 7] 
    for plane_idx, delta in enumerate(queen_directions):
        for dist in range(1, 8):
            plane = plane_idx * 7 + (dist - 1)
            for from_sq in range(64):
                to_sq = from_sq + delta * dist
                
                # Пропускаем ходы, выходящие за доску
                if not (0 <= to_sq < 64):
                    continue
                # Пропускаем ходы с "перескоком" через край доски
                if max(abs(chess.square_file(from_sq) - chess.square_file(to_sq)),
                       abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))) != dist:
                    continue

                index = plane * 64 + from_sq
                move = chess.Move(from_sq, to_sq)
                MOVE_TO_INDEX_MAP[(from_sq, to_sq, None)] = index
                MOVE_TO_INDEX_MAP[(from_sq, to_sq, chess.QUEEN)] = index
                INDEX_TO_MOVE_MAP[index] = move

    # 2. Ходы коня - 8 плоскостей
    knight_plane_start = 56
    knight_deltas = [17, 15, 10, 6, -6, -10, -15, -17]
    for plane_idx, delta in enumerate(knight_deltas):
        plane = knight_plane_start + plane_idx
        for from_sq in range(64):
            to_sq = from_sq + delta
            if not (0 <= to_sq < 64):
                continue
            if max(abs(chess.square_file(from_sq) - chess.square_file(to_sq)),
                   abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))) != 2:
                continue
                
            index = plane * 64 + from_sq
            move = chess.Move(from_sq, to_sq)
            MOVE_TO_INDEX_MAP[(from_sq, to_sq, None)] = index
            INDEX_TO_MOVE_MAP[index] = move

    # 3. Превращения в "слабые" фигуры - 9 плоскостей
    # (3 фигуры * 3 направления)
    promo_plane_start = 64
    promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promo_deltas = {
        chess.WHITE: [7, 8, 9],   # NW, N, NE относительно белых
        chess.BLACK: [-9, -8, -7] # SW, S, SE относительно белых, но вперед для черных
    }
    promo_source_rank = {chess.WHITE: 6, chess.BLACK: 1}
    promo_target_rank = {chess.WHITE: 7, chess.BLACK: 0}

    for piece_idx, piece in enumerate(promo_pieces):
        for delta_idx in range(3):
            plane = promo_plane_start + delta_idx * 3 + piece_idx
            for color in chess.COLORS:
                delta = promo_deltas[color][delta_idx]
                source_rank = promo_source_rank[color]
                target_rank = promo_target_rank[color]

                for from_file in range(8):
                    from_sq = chess.square(from_file, source_rank)
                    to_sq = from_sq + delta
                    if not (0 <= to_sq < 64):
                        continue
                    if chess.square_rank(to_sq) != target_rank:
                        continue
                    if abs(chess.square_file(from_sq) - chess.square_file(to_sq)) > 1:
                        continue

                    index = plane * 64 + from_sq
                    move = chess.Move(from_sq, to_sq, promotion=piece)
                    MOVE_TO_INDEX_MAP[(from_sq, to_sq, piece)] = index
                    INDEX_TO_MOVE_MAP[index] = move

# Запускаем построение карт при импорте модуля
_build_move_maps()

def move_to_index(move: chess.Move):
    """ Конвертирует объект хода chess.Move в индекс политики. """
    return MOVE_TO_INDEX_MAP.get((move.from_square, move.to_square, move.promotion))

def index_to_move(index, board: chess.Board):
    """ Конвертирует индекс политики в объект хода chess.Move. """
    return INDEX_TO_MOVE_MAP.get(index)

# --- НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (перенесены из RL_train.py) ---

def format_board_for_log(board: chess.Board) -> str:
    """
    Создает идеально выровненную ASCII доску с классическими шахматными обозначениями.
    """
    piece_symbols = {
        (chess.PAWN, chess.WHITE): 'P', (chess.PAWN, chess.BLACK): 'p',
        (chess.KNIGHT, chess.WHITE): 'N', (chess.KNIGHT, chess.BLACK): 'n',
        (chess.BISHOP, chess.WHITE): 'B', (chess.BISHOP, chess.BLACK): 'b',
        (chess.ROOK, chess.WHITE): 'R', (chess.ROOK, chess.BLACK): 'r',
        (chess.QUEEN, chess.WHITE): 'Q', (chess.QUEEN, chess.BLACK): 'q',
        (chess.KING, chess.WHITE): 'K', (chess.KING, chess.BLACK): 'k'
    }
    lines = ["  +---+---+---+---+---+---+---+---+"]
    for rank in range(7, -1, -1):
        rank_line = f"{rank + 1} |"
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = piece_symbols.get((piece.piece_type, piece.color), " ") if piece else " "
            rank_line += f" {symbol} |"
        lines.append(rank_line)
        lines.append("  +---+---+---+---+---+---+---+---+")
    lines.append("    a   b   c   d   e   f   g   h")
    return "\n".join(lines)

def format_board_for_html(board: chess.Board) -> str:
    """
    Создает HTML-фрагмент с доской для вставки в файл.
    """
    piece_symbols = {
        (chess.PAWN, chess.WHITE): 'P', (chess.PAWN, chess.BLACK): 'p',
        (chess.KNIGHT, chess.WHITE): 'N', (chess.KNIGHT, chess.BLACK): 'n',
        (chess.BISHOP, chess.WHITE): 'B', (chess.BISHOP, chess.BLACK): 'b',
        (chess.ROOK, chess.WHITE): 'R', (chess.ROOK, chess.BLACK): 'r',
        (chess.QUEEN, chess.WHITE): 'Q', (chess.QUEEN, chess.BLACK): 'q',
        (chess.KING, chess.WHITE): 'K', (chess.KING, chess.BLACK): 'k'
    }
    lines = ['<pre style="font-family: \'Courier New\', Consolas, monospace; font-size: 14px; line-height: 1.2;">']
    lines.append("  +---+---+---+---+---+---+---+---+")
    for rank in range(7, -1, -1):
        rank_line = f"{rank + 1} |"
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = piece_symbols.get((piece.piece_type, piece.color), " ") if piece else " "
            rank_line += f" {symbol} |"
        lines.append(rank_line)
        lines.append("  +---+---+---+---+---+---+---+---+")
    lines.append("    a   b   c   d   e   f   g   h")
    lines.append('</pre>')
    return "\n".join(lines)

def create_html_board_file(board: chess.Board, game_num: int, move_num: int, last_move: str):
    """
    Создает/обновляет HTML файл с текущей доской для просмотра в браузере.
    """
    from rl_chess.config import HTML_UPDATE_EVERY_N_MOVES # Локальный импорт, чтобы избежать циклов
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RL Chess - Игра #{game_num}, Ход #{move_num}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .board {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        pre {{ font-family: 'Courier New', Consolas, monospace; font-size: 16px; line-height: 1.3; margin: 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 RL Chess Training</h1>
        <h2>Игра #{game_num} | Ход #{move_num} | Последний ход: {last_move}</h2>
    </div>
    <div class="board">
        {format_board_for_html(board)}
    </div>
</body>
</html>"""
    with open("current_board.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def get_uci_move_string(move, board):
    """Converts a move to UCI string format, handling null moves."""
    if move is None:
        return "0000"
    return move.uci()

def get_live_logger(log_file, logger_name):
    """Creates a logger that writes to a specific file for live updates."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        # Добавляем миллисекунды (%(msecs)03d) в форматтер для более точного времени
        formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(name)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def setup_worker_logging():
    """
    Настраивает логирование для дочерних процессов.
    В отличие от logging.basicConfig, это можно безопасно вызывать в процессах,
    созданных через 'spawn', чтобы они начали писать в тот же лог-файл.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(processName)s] %(message)s'
    )
    # Направляем лог в тот же основной файл
    file_handler = logging.FileHandler("distributed_training.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    # Получаем корневой логгер и настраиваем его
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Избегаем дублирования хендлеров
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)

def format_move(move):
    """Formats a chess.Move object for pretty printing."""
    if move is None:
        return "NULL"
    return move.uci() 