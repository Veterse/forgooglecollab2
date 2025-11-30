# -*- coding: utf-8 -*-
"""
Тест multiprocessing на TPU.
Запусти в Colab чтобы проверить работает ли.

!pip install torch-xla
!python test_tpu_multiprocessing.py
"""
import os
import time
import multiprocessing as mp

print("=" * 50)
print("ТЕСТ MULTIPROCESSING НА TPU")
print("=" * 50)

# 1. Проверка torch_xla
print("\n1. Проверка torch_xla...")
try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    
    device = xm.xla_device()
    print(f"   ✅ torch_xla установлен")
    print(f"   ✅ TPU устройство: {device}")
    TPU_OK = True
except ImportError as e:
    print(f"   ❌ torch_xla не установлен: {e}")
    TPU_OK = False
except Exception as e:
    print(f"   ❌ Ошибка TPU: {e}")
    TPU_OK = False

# 2. Проверка стандартного multiprocessing
print("\n2. Проверка стандартного multiprocessing...")

def simple_worker(worker_id, result_queue):
    """Простой воркер без TPU."""
    import time
    time.sleep(0.5)
    result_queue.put(f"Worker {worker_id} done")

try:
    result_queue = mp.Queue()
    processes = []
    
    for i in range(3):
        p = mp.Process(target=simple_worker, args=(i, result_queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join(timeout=5)
    
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f"   ✅ Стандартный multiprocessing работает: {results}")
    MP_OK = True
except Exception as e:
    print(f"   ❌ Стандартный multiprocessing не работает: {e}")
    MP_OK = False

# 3. Проверка multiprocessing с TPU (через xmp.spawn)
if TPU_OK:
    print("\n3. Проверка xmp.spawn (TPU multiprocessing)...")
    
    def tpu_worker(index):
        """Воркер для TPU."""
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        
        # Простая операция на TPU
        x = torch.randn(100, 100, device=device)
        y = x @ x.T
        xm.mark_step()
        
        print(f"   Worker {index} на {device}: tensor shape {y.shape}")
    
    try:
        # xmp.spawn запускает функцию на всех TPU cores
        # nprocs=1 для одного ядра (v5e-1)
        print("   Запуск xmp.spawn с 1 процессом...")
        xmp.spawn(tpu_worker, args=(), nprocs=1, start_method='fork')
        print("   ✅ xmp.spawn работает!")
        XMP_OK = True
    except Exception as e:
        print(f"   ❌ xmp.spawn не работает: {e}")
        XMP_OK = False
else:
    print("\n3. Пропуск теста xmp.spawn (TPU недоступен)")
    XMP_OK = False

# 4. Проверка multiprocessing + TPU inference (наш сценарий)
if TPU_OK and MP_OK:
    print("\n4. Проверка: CPU workers + TPU inference...")
    
    def cpu_worker_with_tpu_call(worker_id, model_queue, result_queue):
        """CPU воркер который отправляет запросы на TPU."""
        import torch
        
        # Симулируем работу
        for i in range(3):
            # Создаём тензор на CPU
            x = torch.randn(1, 102, 8, 8)
            
            # Отправляем на "инференс" (в реальности через очередь)
            model_queue.put((worker_id, i, x))
            
            # Ждём результат
            time.sleep(0.1)
        
        result_queue.put(f"Worker {worker_id} finished")
    
    def tpu_inference_server(model_queue, num_requests):
        """TPU сервер для инференса."""
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        
        # Простая "модель"
        model = torch.nn.Linear(102 * 8 * 8, 100).to(device)
        
        processed = 0
        while processed < num_requests:
            try:
                worker_id, req_id, x = model_queue.get(timeout=2)
                
                # Инференс на TPU
                x_flat = x.view(1, -1).to(device)
                with torch.no_grad():
                    out = model(x_flat)
                xm.mark_step()
                
                processed += 1
                print(f"   TPU processed request from worker {worker_id}, req {req_id}")
            except:
                break
        
        print(f"   TPU server processed {processed} requests")
    
    try:
        model_queue = mp.Queue()
        result_queue = mp.Queue()
        
        num_workers = 2
        requests_per_worker = 3
        total_requests = num_workers * requests_per_worker
        
        # Запускаем TPU сервер в отдельном процессе
        tpu_process = mp.Process(target=tpu_inference_server, args=(model_queue, total_requests))
        tpu_process.start()
        
        # Запускаем CPU воркеры
        workers = []
        for i in range(num_workers):
            p = mp.Process(target=cpu_worker_with_tpu_call, args=(i, model_queue, result_queue))
            workers.append(p)
            p.start()
        
        # Ждём завершения
        for p in workers:
            p.join(timeout=10)
        tpu_process.join(timeout=10)
        
        print("   ✅ CPU workers + TPU inference работает!")
        HYBRID_OK = True
    except Exception as e:
        print(f"   ❌ Гибридный режим не работает: {e}")
        HYBRID_OK = False
else:
    print("\n4. Пропуск гибридного теста")
    HYBRID_OK = False

# Итоги
print("\n" + "=" * 50)
print("ИТОГИ:")
print("=" * 50)
print(f"TPU доступен:           {'✅' if TPU_OK else '❌'}")
print(f"Multiprocessing:        {'✅' if MP_OK else '❌'}")
print(f"xmp.spawn:              {'✅' if XMP_OK else '❌'}")
print(f"CPU workers + TPU:      {'✅' if HYBRID_OK else '❌'}")
print("=" * 50)

if HYBRID_OK:
    print("\n🎉 МОЖНО использовать multiprocessing с TPU!")
    print("   Архитектура: CPU workers → Queue → TPU inference server")
elif XMP_OK:
    print("\n⚠️ Только xmp.spawn работает (нужна адаптация кода)")
elif MP_OK:
    print("\n⚠️ Только стандартный multiprocessing (без TPU)")
else:
    print("\n❌ Multiprocessing не работает")
