# -*- coding: utf-8 -*-
import chess
import numpy as np
import torch
import logging
import math
from collections import deque
from . import config
from .RL_network import board_to_tensor, ChessNetwork
from .RL_utils import move_to_index

# TPU support
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Устанавливаем логгер
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # {move: Node}
        self.visit_count = 0
        self.total_action_value = 0.0
        self.prior_p = prior_p
        self.is_expanding = False

    def expand(self, policy_probs):
        for move, prob in policy_probs.items():
            if move not in self.children:
                self.children[move] = Node(parent=self, prior_p=prob)
    
    def select(self, c_puct):
        best_score = -float('inf')
        best_action = None
        best_child = None

        available = []
        blocked = []
        for move, child in self.children.items():
            if child.is_expanding:
                blocked.append((move, child))
            else:
                available.append((move, child))

        candidates = available if available else blocked

        for move, child in candidates:
            score = child.get_puct_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = move
                best_child = child
        return best_action, best_child

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_action_value += value
        if self.parent:
            # Значение всегда с точки зрения текущего игрока,
            # поэтому для родителя (ход другого игрока) его нужно инвертировать
            self.parent.backpropagate(-value)

    def get_puct_score(self, c_puct):
        # Q-значение (средняя ценность действия)
        q_value = self.total_action_value / (self.visit_count + 1e-8)
        # U-значение (бонус за исследование)
        u_value = (c_puct * self.prior_p *
                   math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return q_value + u_value
    
    def is_leaf(self):
        return not self.children

class MCTSAgent:
    def __init__(self, model: ChessNetwork, device, num_simulations=1600, predictor=None, thinking_logger=None, log_mcts=False):
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = 1.5
        self.batch_size = config.MCTS_BATCH_SIZE
        # Параметры для шума Дирихле
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        self.thinking_logger = thinking_logger
        self.log_mcts = log_mcts
        
        if predictor is not None:
            # Если передан предсказатель (например, PredictionClient), используем его
            self.predictor = predictor
            self.model = None
        else:
            # Иначе используем локальную модель (для совместимости)
            self.model = model.to(device)
            self.model.eval()
            self.predictor = self._local_predict

    def _local_predict(self, tensors):
        """Локальный предсказатель для совместимости с обычным режимом."""
        # Определяем тип устройства
        device_type = str(self.device.type) if hasattr(self.device, 'type') else str(self.device)
        is_cuda = 'cuda' in device_type
        is_tpu = 'xla' in device_type
        
        # Autocast только для CUDA
        use_amp = is_cuda
        
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    log_policies, values = self.model(tensors)
            else:
                log_policies, values = self.model(tensors)
        
        # TPU синхронизация
        if is_tpu and TPU_AVAILABLE:
            xm.mark_step()
        
        # Приводим к float32 перед переходом в NumPy
        return log_policies.float(), values.float()

    def _add_dirichlet_noise(self, policy_dict):
        """Добавляет шум Дирихле к априорным вероятностям для исследования."""
        moves = list(policy_dict.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        noisy_policy = {}
        for i, move in enumerate(moves):
            noisy_policy[move] = (1 - self.dirichlet_epsilon) * policy_dict[move] + self.dirichlet_epsilon * noise[i]
        return noisy_policy

    @staticmethod
    def _format_policy_log(policy_dict, top_n=5):
        """Форматирует словарь политики для красивого вывода в лог."""
        if not policy_dict:
            return "  (пусто)"
        sorted_policy = sorted(policy_dict.items(), key=lambda item: item[1], reverse=True)
        return "\n".join([f"  {i+1}. {move.uci()} ({prob:.3f})" for i, (move, prob) in enumerate(sorted_policy[:top_n])])

    @staticmethod
    def _format_mcts_log(root_node, top_n=3):
        """Форматирует результаты MCTS для лога."""
        mcts_stats = []
        for move, node in root_node.children.items():
            if node.visit_count > 0:
                # Q-значение для дочернего узла - это средняя ценность действия с точки зрения родителя (root)
                q_value = node.total_action_value / node.visit_count
                mcts_stats.append((move, node.visit_count, q_value))
        
        if not mcts_stats:
            return "  (нет данных)"

        sorted_mcts = sorted(mcts_stats, key=lambda item: item[1], reverse=True)
        # При выводе Q-значения инвертируем его, чтобы оно было с точки зрения текущего игрока
        return "\n".join([f"  {i+1}. {move.uci()}: {visits} посещений, Q={-q_value:+.3f}" for i, (move, visits, q_value) in enumerate(sorted_mcts[:top_n])])

    def get_move(self, board: chess.Board, board_history=None, temperature: float = 0.1, is_self_play: bool = False):
        """
        Выбирает ход с использованием MCTS.

        Args:
            board: Текущая позиция на доске.
            temperature: Температура для выбора хода. Высокая для исследования, низкая для эксплуатации.
            is_self_play: Флаг, указывающий, используется ли метод в цикле self-play.
                          Если True, добавляется шум Дирихле для исследования.
        """
        # --- Начало игры/хода логируется извне, т.к. агент не знает номера игры/хода ---
        
        root = Node(prior_p=1.0)
        root_history = self._prepare_history(board, board_history)

        # --- 1. Первоначальная оценка и расширение корневого узла ---
        self._evaluate_and_expand([root], [root_history])

        # --- 2. Логирование первоначального мнения сети ---
        clean_policy = {move: node.prior_p for move, node in root.children.items()}
        initial_value = root.total_action_value if root.visit_count > 0 else 0.0

        # Логируем только если есть хоть какие-то легальные ходы
        if clean_policy:
            logger.debug(f"Мнение сети до MCTS:\n"
                         f"  Оценка (Value): {initial_value:.3f}\n"
                         f"  Чистая политика (топ-5):\n{self._format_policy_log(clean_policy)}")

        # --- 3. Добавление шума Дирихле для исследования (только в self-play) ---
        if is_self_play and clean_policy:
            noisy_policy = self._add_dirichlet_noise(clean_policy)
            logger.debug(f"Применен шум к политике в режиме обучения:\n"
                         f"  Зашумленная политика (топ-5):\n{self._format_policy_log(noisy_policy)}")
            
            # Обновляем априорные вероятности в дочерних узлах с учетом шума
            for move, node in root.children.items():
                if move in noisy_policy:
                    node.prior_p = noisy_policy[move]

        # --- 4. Основной цикл MCTS ---
        num_remaining_sims = self.num_simulations - 1
        if num_remaining_sims > 0:
            num_batches = num_remaining_sims // self.batch_size
            remainder = num_remaining_sims % self.batch_size

            def run_batch(batch_size):
                if batch_size <= 0:
                    return
                leaves, histories_to_eval = [], []
                for _ in range(batch_size):
                    leaf, temp_history = self._select_leaf(root, board, root_history)
                    leaves.append(leaf)
                    histories_to_eval.append(temp_history)
                self._evaluate_and_expand(leaves, histories_to_eval)

            for _ in range(num_batches):
                run_batch(self.batch_size)
            if remainder:
                run_batch(remainder)

        # --- 5. Выбор хода и финальное логирование ---
        if clean_policy and self.log_mcts and self.thinking_logger is not None:
            try:
                self.thinking_logger.info(f"Position FEN: {board.fen()}")
                self.thinking_logger.info(f"Network value estimate: {initial_value:.3f}")
                self.thinking_logger.info("Network policy (top-5):")
                self.thinking_logger.info(self._format_policy_log(clean_policy))
                self.thinking_logger.info(f"MCTS ({self.num_simulations} simulations) top-3:")
                self.thinking_logger.info(self._format_mcts_log(root))
            except Exception:
                pass

        if clean_policy:
            logger.debug(f"MCTS ({self.num_simulations} сим.) топ-3:\n{self._format_mcts_log(root)}")

        visit_counts = {move: node.visit_count for move, node in root.children.items()}
        
        if not visit_counts:
            return None, None

        moves, visits = zip(*visit_counts.items())
        total_visits = sum(visits)
        policy_target = torch.zeros(4672, device=self.device)
        for move, v_count in zip(moves, visits):
            idx = move_to_index(move)
            if idx is not None:
                policy_target[idx] = v_count / total_visits

        if not is_self_play or temperature < 1e-3: # "Жадный" выбор для игры или в конце партии
            chosen_move = max(visit_counts, key=visit_counts.get)
        else: # Вероятностный выбор для исследования в self-play
            visit_tensor = torch.tensor([v**(1/temperature) for v in visits], dtype=torch.float32)
            probabilities = visit_tensor / torch.sum(visit_tensor)
            move_idx = torch.multinomial(probabilities, 1).item()
            chosen_move = moves[move_idx]
        
        if self.log_mcts and self.thinking_logger is not None:
            try:
                chosen_visits = visit_counts.get(chosen_move, 0)
                chosen_prob = chosen_visits / total_visits if total_visits > 0 else 0.0
                node = root.children.get(chosen_move)
                q_value = 0.0
                if node is not None and node.visit_count > 0:
                    q_value = node.total_action_value / node.visit_count
                self.thinking_logger.info(f"Chosen move: {chosen_move.uci()} (visits={chosen_visits}, prob={chosen_prob:.3f}, Q={-q_value:+.3f})")
                self.thinking_logger.info("-----")
            except Exception:
                pass

        logger.info(f"Выбран ход: {chosen_move.uci()}")
        return chosen_move, policy_target

    def _select_leaf(self, root_node, root_board, root_history):
        """Спускается по дереву до листового узла."""
        node = root_node
        board = root_board.copy()
        history = deque((b.copy() for b in root_history), maxlen=config.BOARD_HISTORY_LENGTH)
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            board.push(move)
            history.append(board.copy())
        node.is_expanding = True
        return node, history

    def _evaluate_and_expand(self, nodes, histories):
        """Оценивает узел(ы) нейросетью, расширяет их и делает backpropagate."""
        if not nodes:
            return

        tensors = torch.stack([board_to_tensor(history, self.device) for history in histories])

        # Используем абстракцию predictor (это может быть локальная модель или удаленный сервер)
        # Важно: predictor возвращает тензоры на CPU (если это PredictionClient)
        # или на GPU (если это _local_predict), поэтому мы приводим к numpy аккуратно.
        log_policies, values = self.predictor(tensors)

        policies = torch.exp(log_policies).cpu().numpy()
        values = values.cpu().numpy().flatten()

        for i, node in enumerate(nodes):
            history = histories[i]
            board = history[-1]
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                if result == '1-0': value = 1.0
                elif result == '0-1': value = -1.0
                else: value = 0.0
                if board.turn != chess.WHITE:
                    value = -value
            else:
                value = values[i]
                policy_probs = self._get_policy_dict(policies[i], board)
                if node.is_leaf():
                    node.expand(policy_probs)
            
            node.backpropagate(value)
            node.is_expanding = False

    def _get_policy_dict(self, policy_array, board):
        """Преобразует массив вероятностей от сети в словарь {move: prob}."""
        policy_dict = {}
        for move in board.legal_moves:
            idx = move_to_index(move)
            if idx is not None:
                policy_dict[move] = policy_array[idx]
        
        # Нормализация, чтобы сумма вероятностей легальных ходов была 1
        total_prob = sum(policy_dict.values())
        if total_prob > 0:
            for move in policy_dict:
                policy_dict[move] /= total_prob
        return policy_dict

    def _prepare_history(self, board, history):
        max_len = config.BOARD_HISTORY_LENGTH
        if history is None:
            history = deque(maxlen=max_len)
        else:
            history = deque((b.copy() for b in history), maxlen=max_len)

        if not history or history[-1].fen() != board.fen():
            history.append(board.copy())

        return history

       