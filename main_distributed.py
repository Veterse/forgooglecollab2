"""
Главный управляющий скрипт для распределенного обучения.

Этот скрипт инициализирует и запускает все компоненты системы:
- Рабочий-тренер (TrainingWorker)
- Несколько рабочих-игроков (SelfPlayWorker)
- Общую нейросеть и буфер в разделяемой памяти

Он создает необходимые общие объекты и управляет жизненным циклом дочерних процессов.
"""
import multiprocessing
import torch
import logging
import sys
import time
import os
import platform
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# --- Локальные импорты ---
import rl_chess.config as config
from rl_chess.RL_network import ChessNetwork
from rl_chess.shared_buffer import SharedReplayBuffer
from rl_chess.self_play_worker import SelfPlayWorker
from rl_chess.training_worker import TrainingWorker
from rl_chess.inference_server import InferenceServer
import multiprocessing

IS_WINDOWS = platform.system() == "Windows"

def setup_logging():
    """Настраивает глобальное логирование для всех процессов."""
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(processName)s] %(message)s'
    )
    
    # Очищаем лог-файл для live-апдейтов перед стартом
    live_log_file = "live_updates.log"
    if os.path.exists(live_log_file):
        os.remove(live_log_file)
    with open(live_log_file, 'w') as f:
        f.write("Live updates log started.\n")

    # Файловый логгер, в UTF-8, чтобы наверняка сохранить все символы
    file_handler = logging.FileHandler("distributed_training.log", mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)

    # Консольный логгер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Устанавливаем базовую конфигурацию
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def load_checkpoint(model, optimizer, scheduler, training_step_counter, total_games_played_counter, white_wins, black_wins, draws):
    """Загружает чекпоинт, если он существует."""
    if not os.path.exists(config.CHECKPOINT_PATH):
        logging.info("Чекпоинт не найден, начинаем новую сессию обучения.")
        return

    try:
        logging.info(f"Найден чекпоинт: {config.CHECKPOINT_PATH}. Загрузка...")
        device = torch.device(config.TRAINING_DEVICE)
        # Если модель скомпилирована, нужно загружать в "чистую" модель, а компилировать потом
        is_compiled = hasattr(model, '_orig_mod')
        model_to_load = model._orig_mod if is_compiled else model
        
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Загружаем значения в общие счетчики
        with training_step_counter.get_lock():
            training_step_counter.value = checkpoint.get('training_steps', 0)
        with total_games_played_counter.get_lock():
            total_games_played_counter.value = checkpoint.get('games_played', 0)
            
        # Загружаем статистику побед, если она есть
        if 'white_wins' in checkpoint:
            with white_wins.get_lock(): white_wins.value = checkpoint['white_wins']
        if 'black_wins' in checkpoint:
            with black_wins.get_lock(): black_wins.value = checkpoint['black_wins']
        if 'draws' in checkpoint:
            with draws.get_lock(): draws.value = checkpoint['draws']

        logging.info(f"✅ Чекпоинт успешно загружен. Возобновление с шага {training_step_counter.value}, сыграно игр: {total_games_played_counter.value}.")
        logging.info(f"📊 Статистика: Белые: {white_wins.value}, Черные: {black_wins.value}, Ничьи: {draws.value}")

    except Exception as e:
        logging.error(f"❌ Не удалось загрузить чекпоинт: {e}. Начинаем с нуля.")

def main():
    """
    Основная функция для запуска распределенной системы.
    """
    setup_logging()
    logging.info("Starting distributed training system with Shared Model and Memory Buffer...")

    try:
        # На Windows используем spawn, на Linux/WSL оставляем дефолтный метод (fork),
        # чтобы torch.compile можно было безопасно применять без проблем с pickling.
        if IS_WINDOWS:
            multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    total_games_played_counter = multiprocessing.Value('i', 0)
    training_step_counter = multiprocessing.Value('i', 0)
    
    # --- Статистика побед (глобальная) ---
    white_wins = multiprocessing.Value('i', 0)
    black_wins = multiprocessing.Value('i', 0)
    draws = multiprocessing.Value('i', 0)
    
    # --- Очереди для Inference Server ---
    # Общая очередь для входящих запросов от всех воркеров
    input_queue = multiprocessing.Queue()
    
    num_workers = 8  # Количество рабочих-игроков (больше = быстрее генерация)
    # Личные очереди ответов для каждого воркера
    output_queues = [multiprocessing.Queue() for _ in range(num_workers)]
    
    # --- Shared Memory Buffer для Инференса (Zero-Copy) ---
    # Размер: [num_workers, config.MCTS_BATCH_SIZE, config.INPUT_CHANNELS, 8, 8]
    # Каждый воркер пишет в свой слот.
    logging.info(f"Allocating shared memory for inference: {num_workers} workers x {config.MCTS_BATCH_SIZE} batch size...")
    inference_buffer_shape = (num_workers, config.MCTS_BATCH_SIZE, config.INPUT_CHANNELS, 8, 8)
    inference_shared_buffer = torch.zeros(inference_buffer_shape, dtype=torch.float32).share_memory_()
    logging.info("Shared inference buffer allocated.")
    
    replay_buffer = SharedReplayBuffer(config.MAX_REPLAY_BUFFER_SIZE)

    # Глобальные CUDA/cuDNN оптимизации (если тренировка идёт на GPU)
    if config.TRAINING_DEVICE == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    model = ChessNetwork()
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ExponentialLR(optimizer, gamma=config.SCHEDULER_GAMMA)
    
    # Загружаем чекпоинт в чистую модель ДО компиляции
    load_checkpoint(model, optimizer, scheduler, training_step_counter, total_games_played_counter, white_wins, black_wins, draws)

    # На Linux/WSL при наличии CUDA используем torch.compile() для ускорения.
    # На Windows оставляем модель без компиляции из-за ограничений multiprocessing/pickling.
    if (not IS_WINDOWS) and config.TRAINING_DEVICE == 'cuda' and torch.cuda.is_available():
        logging.info("Компиляция модели с помощью torch.compile() (Linux/WSL, ожидается ускорение)...")
        model = torch.compile(model, mode="max-autotune")
    else:
        logging.info("Запуск распределённого обучения без torch.compile() (совместимо с multiprocessing на Windows или без CUDA)...")

    model.share_memory() 
    logging.info("Neural network model has been moved to shared memory.")
    
    # ... остальной код синхронизации шедулера и перемещения на GPU ...
    if training_step_counter.value > 0:
        logging.info(f"Синхронизация LR... Проматываем планировщик на {training_step_counter.value} шагов.")
        temp_scheduler = ExponentialLR(optimizer, gamma=config.SCHEDULER_GAMMA)
        temp_scheduler.load_state_dict(scheduler.state_dict())
        optimizer.param_groups[0]['lr'] = temp_scheduler.get_last_lr()[0]
        scheduler = ExponentialLR(optimizer, gamma=config.SCHEDULER_GAMMA, last_epoch=temp_scheduler.last_epoch)
        logging.info(f"LR синхронизирован. Текущее значение: {optimizer.param_groups[0]['lr']:.8f}")

    if config.TRAINING_DEVICE == 'cuda':
        logging.info("Перемещение состояния оптимизатора на CUDA...")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    processes = []
    try:
        # 1. Запускаем Inference Server (держит модель на GPU)
        inference_server = InferenceServer(
            shared_model_state_dict=model, # Передаем саму модель, сервер скопирует веса
            input_queue=input_queue,
            output_queues=output_queues,
            shared_inference_buffer=inference_shared_buffer
        )
        # Передаем модель в сервер (костыль для multiprocessing на Windows, но работает с shared_memory)
        inference_server.set_model(model) 
        processes.append(inference_server)
        inference_server.start()
        logging.info("Inference Server started.")

        # 2. Запускаем Training Worker (обновляет модель)
        training_worker = TrainingWorker(
            model=model,
            # model_lock убран, так как инференс отделен, а обновления атомарны для shared memory (в теории)
            # либо мы допускаем race condition при чтении весов сервером, что не критично.
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_step_counter=training_step_counter,
            total_games_played_counter=total_games_played_counter,
            stats_counters=(white_wins, black_wins, draws) # Передаем счетчики для сохранения
        )
        processes.append(training_worker)
        training_worker.start()
        logging.info("Training Worker started.")
        
        logging.info(f"Starting {num_workers} self-play worker processes...")
        for i in range(num_workers):
            worker = SelfPlayWorker(
                worker_id=i,
                input_queue=input_queue,
                output_queue=output_queues[i],
                replay_buffer=replay_buffer,
                total_games_played_counter=total_games_played_counter,
                shared_inference_buffer=inference_shared_buffer,
                stats_counters=(white_wins, black_wins, draws) # Передаем счетчики для обновления
            )
            processes.append(worker)
            worker.start()

        for p in processes:
            p.join()

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt signal received. Terminating all processes...")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        logging.info("All processes have been successfully terminated.")

if __name__ == '__main__':
    main()