# -*- coding: utf-8 -*-
"""
Модуль Inference Server.
Обеспечивает централизованное выполнение предсказаний нейросети на GPU.
Собирает запросы от множества воркеров в батчи (Batching) для максимальной утилизации GPU.
"""
import multiprocessing
import torch
import time
import queue
import logging
from collections import namedtuple

import rl_chess.config as config
from rl_chess.RL_network import ChessNetwork

# Определяем структуру для передачи запроса
InferenceRequest = namedtuple('InferenceRequest', ['worker_id', 'batch_size'])

class PredictionClient:
    """
    Клиент для отправки запросов на сервер инференса.
    Используется внутри SelfPlayWorker вместо прямой модели.
    """
    def __init__(self, worker_id, input_queue, output_queue, shared_inference_buffer):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shared_inference_buffer = shared_inference_buffer

    def __call__(self, tensors):
        """
        Позволяет вызывать клиент как функцию: policies, values = client(tensors)
        """
        batch_size = tensors.shape[0]
        
        # Копируем тензоры в выделенный слот shared memory
        # Слот воркера: [worker_id, :batch_size, :, :, :]
        # Обратите внимание: мы предполагаем, что batch_size не превышает размер слота (MCTS_BATCH_SIZE)
        self.shared_inference_buffer[self.worker_id, :batch_size] = tensors
        
        # Отправляем сигнал в очередь (только ID и размер)
        self.input_queue.put(InferenceRequest(self.worker_id, batch_size))
        
        # Блокирующе ждем ответа в своей личной очереди
        policies, values = self.output_queue.get()
        return policies, values

class InferenceServer(multiprocessing.Process):
    """
    Процесс, который держит модель на GPU и обрабатывает запросы батчами.
    """
    def __init__(self, shared_model_state_dict, input_queue, output_queues, shared_inference_buffer):
        super().__init__()
        self.shared_model_state_dict = shared_model_state_dict 
        self.input_model = None 
        
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.shared_inference_buffer = shared_inference_buffer
        self.name = "InferenceServer"
        self.daemon = True # Чтобы процесс умирал вместе с главным
        self.stop_event = multiprocessing.Event()

    def set_model(self, model):
        """Принимает модель из главного процесса (shared memory)"""
        self.input_model = model

    def run(self):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(processName)s] %(message)s',
            handlers=[
                logging.FileHandler("distributed_training.log", mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        device = torch.device(config.TRAINING_DEVICE) # Инференс крутим там же где и тренировку, на мощной GPU
        logging.info(f"🚀 Inference Server запущен на {device}. Ожидание запросов...")

        # Инициализация модели на GPU
        model = ChessNetwork().to(device)
        model.eval()
        
        # Первоначальная синхронизация весов
        if self.input_model:
            model.load_state_dict(self.input_model.state_dict())
            logging.info("Веса модели загружены из shared memory.")
        else:
            logging.warning("Внимание: Входная модель не передана, используются случайные веса!")

        # Подготовка к AMP (Mixed Precision)
        use_amp = (device.type == 'cuda')
        dtype = torch.float16 if use_amp else torch.float32
        if use_amp and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            
        logging.info(f"Режим точности: {'AMP (' + str(dtype) + ')' if use_amp else 'FP32'}")

        # Переменные для цикла
        requests_buffer = []
        last_sync_time = time.time()
        SYNC_INTERVAL = 5.0 # Синхронизировать веса каждые 5 секунд

        while not self.stop_event.is_set():
            # 1. Сбор батча
            start_wait = time.time()
            
            # ДИНАМИЧЕСКИЙ БАТЧИНГ
            # Пытаемся собрать как можно больше запросов, но не меньше 1, если очередь не пуста.
            # Если мы уже собрали INFERENCE_BATCH_SIZE, то отправляем сразу.
            # Если нет, ждем недолго, вдруг прилетит еще.
            
            # Первый запрос ждем с таймаутом (чтобы не крутить цикл впустую)
            try:
                req = self.input_queue.get(timeout=config.INFERENCE_TIMEOUT)
                requests_buffer.append(req)
            except queue.Empty:
                # Если пусто, проверяем синхронизацию и идем на новый круг
                pass

            # Если получили хотя бы один запрос, пробуем добрать еще без ожидания
            if requests_buffer:
                # Вычисляем сколько еще можем взять до оптимального размера (например, 256 или 512 для H100)
                # Для H100 можно смело брать побольше. INFERENCE_BATCH_SIZE в конфиге можно увеличить.
                # Здесь мы просто выгребаем всё что есть в очереди прямо сейчас.
                while len(requests_buffer) < config.INFERENCE_BATCH_SIZE * 2: # *2 как запас для H100
                    try:
                        # non-blocking get
                        req = self.input_queue.get_nowait()
                        requests_buffer.append(req)
                    except queue.Empty:
                        break
            
            if not requests_buffer:
                # Если работы нет, проверим не пора ли обновить веса
                if time.time() - last_sync_time > SYNC_INTERVAL:
                    if self.input_model:
                        # Загружаем веса из разделяемой памяти (это быстро, т.к. копирование из RAM в VRAM)
                        model.load_state_dict(self.input_model.state_dict())
                    last_sync_time = time.time()
                continue

            # 2. Подготовка данных
            # requests_buffer содержит [Request(id, batch_size), ...]
            
            all_tensors = []
            request_sizes = []
            worker_ids = []
            
            for req in requests_buffer:
                # Читаем напрямую из shared memory без копирования (zero-copy view)
                # self.shared_inference_buffer[req.worker_id, :req.batch_size]
                tensor_view = self.shared_inference_buffer[req.worker_id, :req.batch_size]
                all_tensors.append(tensor_view)
                request_sizes.append(req.batch_size)
                worker_ids.append(req.worker_id)
            
            # Объединяем все мини-батчи в один большой батч
            # Используем cat. Так как tensor_view находятся в shared memory (CPU),
            # PyTorch должен эффективно перенести их на GPU.
            full_batch = torch.cat(all_tensors).to(device, non_blocking=True)
            
            # 3. Инференс
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    log_policies, values = model(full_batch)
            
            # Переводим в float32 и на CPU для отправки
            log_policies = log_policies.float().cpu()
            values = values.float().cpu()
            
            # 4. Рассылка ответов
            current_idx = 0
            for i, size in enumerate(request_sizes):
                # Вырезаем кусок, соответствующий запросу
                worker_policy = log_policies[current_idx : current_idx + size]
                worker_value = values[current_idx : current_idx + size]
                current_idx += size
                
                worker_id = worker_ids[i]
                
                # Отправляем результат в личную очередь воркера
                self.output_queues[worker_id].put((worker_policy, worker_value))
            
            requests_buffer.clear()

            # Периодическая синхронизация весов (даже если идет активная работа)
            if time.time() - last_sync_time > SYNC_INTERVAL:
                if self.input_model:
                    model.load_state_dict(self.input_model.state_dict())
                last_sync_time = time.time()
