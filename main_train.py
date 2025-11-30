# -*- coding: utf-8 -*-
"""
Универсальный скрипт обучения с поддержкой TPU, CUDA и CPU.
Автоматически определяет лучшее доступное устройство.

Запуск:
    python main_train.py
    python main_train.py --workers 4 --mcts 200
"""
import os
import sys
import time
import logging
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import torch.optim as optim
import chess
import numpy as np

# TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent
from rl_chess.RL_utils import move_to_index
import rl_chess.config as config

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_device():
    """Определяет устройство: TPU > CUDA > CPU."""
    if TPU_AVAILABLE:
        try:
            device = xm.xla_device()
            logger.info(f"✅ TPU доступен: {device}")
            return device, 'tpu'
        except Exception as e:
            logger.warning(f"TPU ошибка: {e}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"✅ CUDA доступен: {torch.cuda.get_device_name(0)}")
        return device, 'cuda'
    
    logger.info("⚠️ Используется CPU")
    return torch.device('cpu'), 'cpu'


def play_game(model, device, mcts_sims=100):
    """Играет одну партию self-play с полноценным MCTS."""
    agent = MCTSAgent(model, device=device, num_simulations=mcts_sims)
    
    board = chess.Board()
    history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
    game_data = []
    move_count = 0
    
    while not board.is_game_over(claim_draw=True):
        move_count += 1
        
        # Температура
        if move_count <= 15:
            temp = 1.2
        elif move_count <= 30:
            temp = 0.5
        else:
            temp = 0.1
        
        move, policy = agent.get_move(board, board_history=history, temperature=temp, is_self_play=True)
        
        if move is None:
            break
        
        state = board_to_tensor(history, device).cpu()
        policy_cpu = policy.cpu() if policy is not None else torch.zeros(4672)
        game_data.append([state, policy_cpu])
        
        board.push(move)
        history.append(board.copy())
    
    # Результат
    result = board.result(claim_draw=True)
    value = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    
    # Добавляем value
    final_data = []
    for i, (state, policy) in enumerate(game_data):
        v = value if i % 2 == 0 else -value
        final_data.append((state, policy, v))
    
    return final_data, result, move_count


def train_step(model, optimizer, states, policies, values, device, device_type):
    """Один шаг обучения."""
    model.train()
    
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)
    
    optimizer.zero_grad()
    
    log_policy, value_pred = model(states)
    
    value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), values)
    policy_loss = -torch.sum(policies * log_policy) / states.size(0)
    loss = value_loss + policy_loss
    
    loss.backward()
    
    # TPU-специфичный шаг
    if device_type == 'tpu':
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()
    
    return loss.item(), value_loss.item(), policy_loss.item()


def save_checkpoint(model, optimizer, game_num, step, replay_buffer, stats, path="rl_checkpoint.pth"):
    """Сохраняет чекпоинт."""
    # Для TPU переносим на CPU
    if TPU_AVAILABLE:
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'game_number': game_num,
        'training_step': step,
        'replay_buffer': list(replay_buffer)[-20000:],
        'stats': stats
    }
    
    # Атомарное сохранение
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
    
    # Сохраняем модель отдельно
    torch.save(model_state, config.MODEL_PATH)
    
    logger.info(f"💾 Чекпоинт сохранён: игра {game_num}, шаг {step}")


def load_checkpoint(model, optimizer, path="rl_checkpoint.pth"):
    """Загружает чекпоинт."""
    if not os.path.exists(path):
        return 0, 0, deque(maxlen=config.MEMORY_SIZE), {'white': 0, 'black': 0, 'draw': 0}
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        replay_buffer = deque(checkpoint.get('replay_buffer', []), maxlen=config.MEMORY_SIZE)
        stats = checkpoint.get('stats', {'white': 0, 'black': 0, 'draw': 0})
        
        logger.info(f"✅ Чекпоинт загружен: игра {checkpoint['game_number']}, шаг {checkpoint['training_step']}")
        return checkpoint['game_number'], checkpoint['training_step'], replay_buffer, stats
    except Exception as e:
        logger.error(f"Ошибка загрузки чекпоинта: {e}")
        return 0, 0, deque(maxlen=config.MEMORY_SIZE), {'white': 0, 'black': 0, 'draw': 0}



def main():
    parser = argparse.ArgumentParser(description="Универсальное обучение RL Chess")
    parser.add_argument("--workers", type=int, default=4, help="Число параллельных игр (threads)")
    parser.add_argument("--mcts", type=int, default=150, help="MCTS симуляций")
    parser.add_argument("--batch", type=int, default=256, help="Размер батча")
    parser.add_argument("--games-per-train", type=int, default=5, help="Игр между обучениями")
    parser.add_argument("--save-every", type=int, default=20, help="Сохранять каждые N игр")
    parser.add_argument("--max-games", type=int, default=100000, help="Максимум игр")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК ОБУЧЕНИЯ RL CHESS")
    logger.info("=" * 60)
    
    # Устройство
    device, device_type = get_device()
    logger.info(f"Устройство: {device} ({device_type})")
    logger.info(f"Воркеров: {args.workers}, MCTS: {args.mcts}, Batch: {args.batch}")
    
    # Модель
    model = ChessNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Загрузка чекпоинта
    game_num, train_step, replay_buffer, stats = load_checkpoint(model, optimizer)
    
    logger.info(f"Старт с игры #{game_num + 1}, буфер: {len(replay_buffer)}")
    logger.info(f"Статистика: +{stats['white']} -{stats['black']} ={stats['draw']}")
    
    start_time = time.time()
    total_moves = 0
    
    # Lock для потокобезопасности
    model_lock = threading.Lock()
    buffer_lock = threading.Lock()
    
    def play_one_game():
        """Играет одну игру (для ThreadPoolExecutor)."""
        with model_lock:
            model.eval()
        
        data, result, moves = play_game(model, device, mcts_sims=args.mcts)
        return data, result, moves
    
    try:
        while game_num < args.max_games:
            # === ФАЗА 1: Self-Play (параллельно через threads) ===
            games_batch = []
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(play_one_game) for _ in range(args.games_per_train)]
                
                for future in as_completed(futures):
                    try:
                        data, result, moves = future.result()
                        games_batch.append((data, result, moves))
                    except Exception as e:
                        logger.error(f"Ошибка в игре: {e}")
            
            # Обрабатываем результаты
            for data, result, moves in games_batch:
                game_num += 1
                total_moves += moves
                
                # Статистика
                if result == "1-0":
                    stats['white'] += 1
                elif result == "0-1":
                    stats['black'] += 1
                else:
                    stats['draw'] += 1
                
                # Добавляем в буфер
                with buffer_lock:
                    replay_buffer.extend(data)
                
                logger.info(f"Игра #{game_num}: {result} ({moves} ходов) | "
                           f"Буфер: {len(replay_buffer)} | "
                           f"+{stats['white']} -{stats['black']} ={stats['draw']}")
            
            # === ФАЗА 2: Обучение ===
            if len(replay_buffer) >= config.MIN_REPLAY_BUFFER_SIZE:
                model.train()
                
                # Несколько шагов обучения
                num_train_steps = max(1, len(games_batch))
                
                for _ in range(num_train_steps):
                    with buffer_lock:
                        indices = np.random.choice(len(replay_buffer), args.batch, replace=False)
                        batch_data = [replay_buffer[i] for i in indices]
                    
                    states = torch.stack([d[0] for d in batch_data])
                    policies = torch.stack([d[1] for d in batch_data])
                    values = torch.tensor([d[2] for d in batch_data], dtype=torch.float32)
                    
                    loss, v_loss, p_loss = train_step(
                        model, optimizer, states, policies, values, device, device_type
                    )
                    train_step_num = train_step + 1
                    train_step = train_step_num
                    
                    if train_step % 10 == 0:
                        logger.info(f"Шаг {train_step} | Loss: {loss:.4f} (v:{v_loss:.4f} p:{p_loss:.4f})")
            
            # === Сохранение ===
            if game_num % args.save_every == 0:
                save_checkpoint(model, optimizer, game_num, train_step, replay_buffer, stats)
                
                elapsed = time.time() - start_time
                games_per_hour = game_num / elapsed * 3600 if elapsed > 0 else 0
                avg_moves = total_moves / game_num if game_num > 0 else 0
                
                logger.info(f"📊 {games_per_hour:.1f} игр/час | Средняя длина: {avg_moves:.0f}")
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Прервано пользователем")
    
    finally:
        save_checkpoint(model, optimizer, game_num, train_step, replay_buffer, stats)
        logger.info(f"\n{'='*60}")
        logger.info(f"ИТОГО: {game_num} игр, {train_step} шагов")
        logger.info(f"Счёт: +{stats['white']} -{stats['black']} ={stats['draw']}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
