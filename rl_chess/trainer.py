# -*- coding: utf-8 -*-
"""
Модуль, отвечающий за процесс обучения нейронной сети.

Содержит функцию `update_network`, которая выполняет один шаг
обновления весов на основе батча данных из буфера воспроизведения.
Эта функция будет вызываться процессом "Тренер".
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging

# Импортируем конфиг напрямую, т.к. параметры обучения здесь критичны
import rl_chess.config as config

def update_network(net, optimizer, scheduler, memory, device, scaler=None):
    """
    Выполняет один шаг обновления весов нейросети.
    Эта функция является сердцем процесса "Тренера".
    """
    net.train()
    
    # Собираем батч случайных данных из памяти
    indices = np.random.choice(len(memory), config.BATCH_SIZE, replace=False)
    batch = [memory[i] for i in indices]
    
    states, policy_targets, value_targets = zip(*batch)
    
    states = torch.stack(states).to(device)
    policy_targets = torch.stack(policy_targets).to(device)
    value_targets = torch.tensor(value_targets, dtype=torch.float32).view(-1, 1).to(device)
    
    total_loss = 0
    
    # Цикл по эпохам
    optimizer.zero_grad() # Обнуляем градиенты перед началом эпох
    for epoch in range(config.EPOCHS_PER_UPDATE):
        # Перемешиваем данные для каждой эпохи для лучшего обучения
        permutation = torch.randperm(states.size(0))
        states_shuffled = states[permutation]
        policy_targets_shuffled = policy_targets[permutation]
        value_targets_shuffled = value_targets[permutation]
        
        # Цикл по мини-батчам с накоплением градиентов
        for i in range(0, config.BATCH_SIZE, config.BATCH_SIZE // config.GRADIENT_ACCUMULATION_STEPS):
            start_idx = i
            end_idx = start_idx + (config.BATCH_SIZE // config.GRADIENT_ACCUMULATION_STEPS)
            
            states_batch = states_shuffled[start_idx:end_idx]
            policy_targets_batch = policy_targets_shuffled[start_idx:end_idx]
            value_targets_batch = value_targets_shuffled[start_idx:end_idx]

            # Используем autocast для Mixed Precision Training
            # with torch.cuda.amp.autocast():
            log_policy_preds, value_preds = net(states_batch)
            
            # Функция потерь:
            # 1. Потери значения (Value Loss) - Mean Squared Error
            value_loss = F.mse_loss(value_preds, value_targets_batch)
            
            # 2. Потери политики (Policy Loss) - Cross-Entropy
            policy_loss = -torch.sum(policy_targets_batch * log_policy_preds) / states_batch.size(0)

            loss = value_loss + policy_loss

            # Масштабируем потери для каждого шага накопления
            scaled_loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            # Вычисляем градиенты
            # scaler.scale(scaled_loss).backward()
            scaled_loss.backward()
            
            total_loss += loss.item()
    
    # Обновляем веса один раз после всех шагов накопления
    # scaler.step(optimizer)
    optimizer.step()
    
    scheduler.step()
    # scaler.update()
    optimizer.zero_grad() # Убедимся, что градиенты снова сброшены
    
    avg_loss = total_loss / (config.EPOCHS_PER_UPDATE * config.GRADIENT_ACCUMULATION_STEPS)
    logging.info(f"   - Средние потери (Loss) за обновление: {avg_loss:.4f}") 