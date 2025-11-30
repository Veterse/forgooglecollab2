import chess
import numpy as np

# --- Piece-Square Tables (from White's perspective) ---
# Values are in centipawns, will be normalized later.
# Source: Simplified from various chess programming wikis.

pawn_pst = np.array([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
])

knight_pst = np.array([
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
])

bishop_pst = np.array([
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
])

rook_pst = np.array([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
])

queen_pst = np.array([
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
])

king_pst = np.array([
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
])

# We store the tables in a dictionary for easy access
piece_psts = {
    chess.PAWN: pawn_pst,
    chess.KNIGHT: knight_pst,
    chess.BISHOP: bishop_pst,
    chess.ROOK: rook_pst,
    chess.QUEEN: queen_pst,
    chess.KING: king_pst
}

class SimpleAI:
    def __init__(self, weights):
        self.weights = weights

    def evaluate_board(self, board):
        if board.is_checkmate():
            return -np.inf if board.turn == chess.WHITE else np.inf
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0
        
        if board.can_claim_threefold_repetition():
            return 0

        # 1. Material score
        material_score = self._calculate_material(board)
        
        # 2. Positional score
        positional_score = self._calculate_positional(board)

        # 3. Pawn structure score
        pawn_structure_score = self._calculate_pawn_structure(board)

        # 4. King safety score
        king_safety_score = self._calculate_king_safety(board)

        # Combine scores using weights
        total_score = (self.weights.get('material', 1.0) * material_score +
                       self.weights.get('position', 0.0) * positional_score +
                       self.weights.get('pawn_structure', 0.0) * pawn_structure_score +
                       self.weights.get('king_safety', 0.0) * king_safety_score)
        
        return total_score if board.turn == chess.WHITE else -total_score

    def _calculate_material(self, board):
        score = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            score += len(board.pieces(piece_type, chess.WHITE)) * self._get_piece_value(piece_type)
            score -= len(board.pieces(piece_type, chess.BLACK)) * self._get_piece_value(piece_type)
        return score

    def _calculate_positional(self, board):
        score = 0
        for piece_type, pst in piece_psts.items():
            # White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                # PSTs are defined from white's perspective, so we read directly
                rank, file = chess.square_rank(square), chess.square_file(square)
                score += pst[7 - rank][file] # np array is indexed from top-left
            
            # Black pieces
            for square in board.pieces(piece_type, chess.BLACK):
                # For black, we flip the rank to read from the same table
                rank, file = chess.square_rank(square), chess.square_file(square)
                score -= pst[rank][file] # The table is already flipped for black perspective
        
        return score / 100.0 # Normalize to be comparable to material score

    def _calculate_pawn_structure(self, board):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            pawns_on_file = [0] * 8
            for sq in board.pieces(chess.PAWN, color):
                pawns_on_file[chess.square_file(sq)] += 1
            
            doubled_pawns_penalty = 0
            for count in pawns_on_file:
                if count > 1:
                    doubled_pawns_penalty += (count - 1)
            
            if color == chess.WHITE:
                score -= doubled_pawns_penalty
            else:
                score += doubled_pawns_penalty
        return score

    def _calculate_king_safety(self, board):
        score = 0
        # Итерация по обоим цветам
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue

            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            pawn_shield_score = 0
            # Проверка пешечного щита перед королем
            pawn_shield_files = range(max(0, king_file - 1), min(7, king_file + 2))
            for file_index in pawn_shield_files:
                # Ищем свои пешки на 1 и 2 ряда впереди
                for rank_offset in [1, 2]: 
                    pawn_rank = king_rank + (rank_offset if color == chess.WHITE else -rank_offset)
                    if 0 <= pawn_rank < 8:
                        pawn_square = chess.square(file_index, pawn_rank)
                        if board.piece_at(pawn_square) == chess.Piece(chess.PAWN, color):
                            pawn_shield_score += 0.25

            # Штраф за атакованные поля вокруг короля
            attack_penalty = 0
            opponent_color = not color
            # Проверяем зону 3x3 вокруг короля
            for r_offset in [-1, 0, 1]:
                for f_offset in [-1, 0, 1]:
                    if r_offset == 0 and f_offset == 0:
                        continue
                    
                    target_rank = king_rank + r_offset
                    target_file = king_file + f_offset

                    if 0 <= target_rank < 8 and 0 <= target_file < 8:
                        target_square = chess.square(target_file, target_rank)
                        if board.is_attacked_by(opponent_color, target_square):
                            attack_penalty += 0.1

            # Итоговый счет безопасности для этого короля
            king_safety_value = pawn_shield_score - attack_penalty

            if color == chess.WHITE:
                score += king_safety_value
            else:
                score -= king_safety_value
                
        return score

    def _get_piece_value(self, piece_type):
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.2,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        return values.get(piece_type, 0)

    def _score_move(self, move, board):
        """Дает оценку ходу для последующей сортировки."""
        # Самый высокий приоритет - у взятий
        if board.is_capture(move):
            # Чем ценнее съеденная фигура и дешевле атакующая, тем лучше ход
            return 10 * self._get_piece_value(board.piece_type_at(move.to_square)) - self._get_piece_value(board.piece_type_at(move.from_square))
        # Затем ходы, которые дают шах
        elif board.gives_check(move):
            return 9
        # Остальные ходы имеют низкий приоритет
        else:
            return 0

    def choose_move(self, board, depth=2):
        best_move = None
        max_eval = -np.inf

        # Сортируем ходы перед началом поиска
        sorted_moves = sorted(board.legal_moves, key=lambda move: self._score_move(move, board), reverse=True)

        for move in sorted_moves:
            board.push(move)
            evaluation = -self._minimax(board, depth - 1, -np.inf, np.inf)
            board.pop()
            
            # Если это первый проанализированный ход или он лучше текущего лучшего, выбираем его
            if best_move is None or evaluation > max_eval:
                max_eval = evaluation
                best_move = move
        
        return best_move

    def _minimax(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        max_eval = -np.inf
        
        # Здесь тоже нужна сортировка для эффективности на следующих уровнях дерева
        sorted_moves = sorted(board.legal_moves, key=lambda move: self._score_move(move, board), reverse=True)

        for move in sorted_moves:
            board.push(move)
            evaluation = -self._minimax(board, depth - 1, -beta, -alpha)
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval