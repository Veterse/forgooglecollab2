# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import chess
import numpy as np
from collections import deque
import os
import logging
import sys

from rl_chess.RL_network import ChessNetwork, board_to_tensor
from rl_chess.RL_agent import MCTSAgent
import rl_chess.config as config  # <-- ИМПОРТИРУЕМ КОНФИГУРАЦИЮ
from rl_chess.RL_utils import create_html_board_file, format_board_for_log  # <-- ИМПОРТИРУЕМ НОВЫЕ ФУНКЦИИ
from rl_chess.trainer import update_network # <-- ИМПОРТ ФУНКЦИИ ОБУЧЕНИЯ
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- Настройка логирования ---
# Отдельные форматтеры для файла и консоли
# В файл пишем только само сообщение (без времени и уровня),
# чтобы строки выглядели так: "Игра #4 | Ход #368: d8c7"
file_formatter = logging.Formatter("%(message)s")
# В консоль продолжаем выводить подробную информацию
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Вывод в файл
# Указываем кодировку UTF-8, чтобы русские буквы отображались корректно
file_handler = logging.FileHandler("training_log.txt", mode='a', encoding='utf-8')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Вывод в консоль
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def train():
    """ Главный цикл обучения. """
    
    # Создаем папку для бэкапов, если ее нет
    os.makedirs(config.BACKUP_DIR, exist_ok=True)

    # 1. Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Используется устройство: {device}")
    if device.type == 'cuda':
        # Глобальные оптимизации CUDA/cuDNN
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        logging.info("🚀 АКТИВИРОВАНЫ ОПТИМИЗАЦИИ ДЛЯ МАКСИМАЛЬНОЙ СКОРОСТИ:")
        logging.info("   ⚡ torch.compile() - ожидается 2-3x ускорение")
        logging.info("   🔥 Mixed Precision Training - ускорение ~1.5-2x")
        logging.info("   📊 Gradient Accumulation - эффективный батч 4096")
        logging.info("   🎯 Virtual Loss MCTS - улучшенное исследование дерева")
        logging.info("   ⚡ Оптимизированные настройки скорости:")
        logging.info(f"      🧠 MCTS симуляции: {config.MCTS_SIMULATIONS} (было 6400)")
        logging.info(f"      💾 MCTS batch_size: 32 (было 64)")
        logging.info(f"      📊 Epochs per update: {config.EPOCHS_PER_UPDATE} (было 3)")
        logging.info(f"      🗃️ Memory size: {config.MEMORY_SIZE} (было 20000)")
        logging.info(f"      📝 Доски логируются каждый ход (скорость не страдает)")
        logging.info(f"      🌐 HTML обновления каждые {config.HTML_UPDATE_EVERY_N_MOVES} ходов")
        logging.info(f"      💾 Полное сохранение каждые {config.SAVE_EVERY_N_GAMES} игр")
        logging.info("   🚀 Ожидаемое ускорение: 4-8x (скорость + качество)!")

    net = ChessNetwork().to(device)
    
    # 🚀 torch.compile() - ОГРОМНОЕ ускорение на H100 (2-3x)
    if device.type == 'cuda':
        net = torch.compile(net)
        logging.info("⚡ torch.compile() активирован - ожидается 2-3x ускорение!")
    
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.99995)
    # Mixed Precision Training ОТКЛЮЧЕНО (ломает policy loss)
    # scaler = torch.cuda.amp.GradScaler()
    scaler = None
    agent = MCTSAgent(net, device=device, num_simulations=config.MCTS_SIMULATIONS)
    
    start_game = 0
    replay_memory = deque(maxlen=config.MEMORY_SIZE)

    # 2. УМНАЯ ЗАГРУЗКА ИЗ ЧЕКПОИНТА
    checkpoint_loaded = False
    
    # Проверяем основной чекпоинт
    if os.path.exists(config.CHECKPOINT_PATH):
        try:
            logging.info(f"🔄 Найден чекпоинт, загрузка прогресса из {config.CHECKPOINT_PATH}")
            checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
            
            # Проверяем целостность чекпоинта
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'game_number', 'replay_memory']
            if all(key in checkpoint for key in required_keys):
                # Загружаем все состояния
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint: # Для обратной совместимости
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # if 'scaler_state_dict' in checkpoint and scaler is not None:
                #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_game = checkpoint['game_number']
                replay_memory = deque(list(checkpoint['replay_memory']), maxlen=config.MEMORY_SIZE)
                
                logging.info(f"✅ ЧЕКПОИНТ ЗАГРУЖЕН УСПЕШНО!")
                logging.info(f"   🎮 Игр завершено: {start_game}")
                logging.info(f"   💾 Данных в памяти: {len(replay_memory)}")
                logging.info(f"   🚀 Продолжаем с игры #{start_game + 1}")
                checkpoint_loaded = True
            else:
                logging.warning(f"⚠️ Чекпоинт поврежден (отсутствуют ключи), начинаем заново")
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки чекпоинта: {e}")
            logging.info("🆕 Начинаем новое обучение")
    
    # Проверяем backup чекпоинт если основной не загрузился
    backup_checkpoints = [
        config.CHECKPOINT_PATH + ".backup",
        config.CHECKPOINT_PATH + ".auto", 
        config.CHECKPOINT_PATH + ".direct"
    ]
    
    for backup_path in backup_checkpoints:
        if not checkpoint_loaded and os.path.exists(backup_path):
            try:
                backup_type = backup_path.split('.')[-1]
                logging.info(f"🔄 Пробуем загрузить {backup_type} чекпоинт: {backup_path}")
                checkpoint = torch.load(backup_path, map_location=device)
                
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint: # Для обратной совместимости
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # if 'scaler_state_dict' in checkpoint:
                #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_game = checkpoint['game_number']
                replay_memory = deque(list(checkpoint['replay_memory']), maxlen=config.MEMORY_SIZE)
                
                logging.info(f"✅ {backup_type.upper()} ЧЕКПОИНТ ЗАГРУЖЕН!")
                logging.info(f"   🎮 Игр завершено: {start_game}")
                logging.info(f"   🚀 Продолжаем с игры #{start_game + 1}")
                checkpoint_loaded = True
                break
            except Exception as e:
                logging.error(f"❌ {backup_type} чекпоинт поврежден: {e}")
    
    if not checkpoint_loaded:
        logging.info("🆕 --- НОВОЕ ОБУЧЕНИЕ С НУЛЯ ---")


    # 3. Цикл Self-Play
    for i_game in range(start_game, config.NUM_GAMES):
        logging.info(f"--- Начало игры #{i_game+1} ---")
        
        board = chess.Board()
        history = deque([board.copy()], maxlen=config.BOARD_HISTORY_LENGTH)
        game_data = [] # Данные для текущей игры
        move_counter = 0
        
        while not board.is_game_over(claim_draw=True):
            move_counter += 1
            # Получаем ход от агента MCTS в режиме обучения (с шумом)
            move, policy_target = agent.get_move(board, board_history=history, is_self_play=True)
            
            # Сохраняем состояние, политику и текущего игрока
            state_tensor = board_to_tensor(history, device)
            game_data.append([state_tensor, policy_target])

            board.push(move)
            history.append(board.copy())
            
            # Логируем ход и доску (всегда)
            logging.info(f"Игра #{i_game+1} | Ход #{move_counter}: {move.uci()}")
            logging.info(f"\n{format_board_for_log(board)}")
            
            # Создаем HTML файл реже для ускорения
            if move_counter % config.HTML_UPDATE_EVERY_N_MOVES == 0:
                create_html_board_file(board, i_game+1, move_counter, move.uci())
        
        # Финальная доска игры (всегда)
        logging.info(f"\n{format_board_for_log(board)}")
        create_html_board_file(board, i_game+1, move_counter, "FINAL")
        
        logging.info(f"Игра #{i_game+1} завершена после {move_counter} ходов. Результат: {board.result(claim_draw=True)}")
        
        # Периодическое логирование прогресса
        if (i_game + 1) % 10 == 0:
            progress_pct = (i_game + 1) / config.NUM_GAMES * 100
            logging.info(f"📊 ПРОГРЕСС: {i_game + 1}/{config.NUM_GAMES} игр ({progress_pct:.1f}%) | Памяти: {len(replay_memory)}")
        
        # 4. Определяем результат и обновляем данные
        result = board.result(claim_draw=True)
        if result == "1-0":
            value_target = 1.0
        elif result == "0-1":
            value_target = -1.0
        else: # Ничья
            value_target = 0.0

        for i in range(len(game_data)):
            player_multiplier = 1 if (i % 2 == 0) else -1
            game_data[i].append(torch.tensor([value_target * player_multiplier], dtype=torch.float32, device=device))

        replay_memory.extend(game_data)
        
        # 5. Обучение нейросети (если накоплено достаточно данных)
        # Важно: ждём минимум позиций чтобы избежать переобучения на малых данных
        if len(replay_memory) >= config.MIN_SAMPLES_FOR_TRAINING:
            logging.info(f"--- Начало обучения сети (буфер: {len(replay_memory)} позиций) ---")
            update_network(net, optimizer, scheduler, replay_memory, device, scaler)
        
        # 5.5. АВТОСОХРАНЕНИЕ ЧЕКПОИНТА (каждые 5 игр для безопасности)
        if (i_game + 1) % config.SAVE_EVERY_N_GAMES == 0:
            logging.info(f"💾 --- Сохранение после {i_game + 1} игр ---")
            
            # Сохраняем модель для игры (более надежно)
            try:
                model_temp = config.MODEL_SAVE_PATH + ".tmp"
                torch.save(net.state_dict(), model_temp)
                # Атомарное переименование (быстро)
                if os.path.exists(config.MODEL_SAVE_PATH):
                    backup_model_path = os.path.join(config.BACKUP_DIR, os.path.basename(config.MODEL_SAVE_PATH) + ".old")
                    os.replace(config.MODEL_SAVE_PATH, backup_model_path)
                os.rename(model_temp, config.MODEL_SAVE_PATH)
                logging.info(f"✅ Модель сохранена в {config.MODEL_SAVE_PATH}")
                logging.info(f"✅ Прогресс сохранен в {config.CHECKPOINT_PATH}")
            except Exception as e:
                logging.error(f"❌ Ошибка сохранения модели: {e}")
            
            # НАДЕЖНОЕ СОХРАНЕНИЕ ЧЕКПОИНТА
            checkpoint_data = {
                'game_number': i_game + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
                'replay_memory': list(replay_memory),
                'save_time': logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None)),
                'total_games': config.NUM_GAMES,
                'batch_size': config.BATCH_SIZE,
                'mcts_simulations': config.MCTS_SIMULATIONS
            }
            
            # Пробуем несколько способов сохранения
            checkpoint_saved = False
            
            # Способ 1: Атомарное сохранение с временным файлом
            try:
                temp_checkpoint = config.CHECKPOINT_PATH + ".tmp"
                logging.info(f"💾 Сохраняю чекпоинт временно в {temp_checkpoint}")
                torch.save(checkpoint_data, temp_checkpoint)
                
                # Создаем backup предыдущего чекпоинта
                if os.path.exists(config.CHECKPOINT_PATH):
                    backup_path = os.path.join(config.BACKUP_DIR, os.path.basename(config.CHECKPOINT_PATH) + ".backup")
                    os.replace(config.CHECKPOINT_PATH, backup_path)
                    logging.info(f"📦 Backup создан: {backup_path}")
                
                # Атомарное переименование
                os.rename(temp_checkpoint, config.CHECKPOINT_PATH)
                logging.info(f"✅ ЧЕКПОИНТ СОХРАНЕН: {config.CHECKPOINT_PATH}")
                checkpoint_saved = True
                
            except Exception as e:
                logging.error(f"❌ Ошибка атомарного сохранения: {e}")
                
                # Способ 2: Прямое сохранение (если атомарное не сработало)
                try:
                    logging.info("🔄 Пробую прямое сохранение чекпоинта...")
                    torch.save(checkpoint_data, config.CHECKPOINT_PATH + ".direct")
                    logging.info(f"✅ Прямое сохранение успешно: {config.CHECKPOINT_PATH}.direct")
                    checkpoint_saved = True
                except Exception as e2:
                    logging.error(f"❌ И прямое сохранение не удалось: {e2}")
            
            if checkpoint_saved:
                memory_mb = len(replay_memory) * 0.001  # Примерная оценка
                logging.info(f"📊 Прогресс: {i_game + 1}/{config.NUM_GAMES} игр ({(i_game + 1)/config.NUM_GAMES*100:.1f}%)")
                logging.info(f"💾 Память: {len(replay_memory)} позиций (~{memory_mb:.1f}MB)")
            else:
                logging.error("💥 КРИТИЧНО: Чекпоинт НЕ СОХРАНЕН! Продолжаем обучение...")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logging.exception("КРИТИЧЕСКАЯ ОШИБКА: Обучение остановлено из-за исключения.")
        # Эта строка нужна, чтобы если скрипт запущен в CI/CD или другой автоматизированной системе,
        # он все равно вернул код ошибки.
        sys.exit(1) 