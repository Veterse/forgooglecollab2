# -*- coding: utf-8 -*-
"""
Парсер distributed_training.log для анализа статистики игр.
"""
import re
import sys

def parse_log(log_file="distributed_training.log"):
    """Парсит лог и выводит статистику."""
    
    # Паттерн для строки с завершением игры
    pattern = r"Игра #(\d+) завершена\. Результат: ([^\s]+) \((\d+) ходов\)"
    
    games = []
    wins_white = 0
    wins_black = 0
    draws = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    game_num = int(match.group(1))
                    result = match.group(2)
                    moves = int(match.group(3))
                    
                    games.append({
                        'num': game_num,
                        'result': result,
                        'moves': moves
                    })
                    
                    if result == "1-0":
                        wins_white += 1
                    elif result == "0-1":
                        wins_black += 1
                    else:
                        draws += 1
    
    except FileNotFoundError:
        print(f"Файл {log_file} не найден!")
        return
    
    if not games:
        print("Игры не найдены в логе.")
        return
    
    # Статистика
    total_games = len(games)
    total_moves = sum(g['moves'] for g in games)
    avg_moves = total_moves / total_games
    min_moves = min(g['moves'] for g in games)
    max_moves = max(g['moves'] for g in games)
    
    # Последние 10 игр
    last_10 = games[-10:] if len(games) >= 10 else games
    avg_last_10 = sum(g['moves'] for g in last_10) / len(last_10)
    
    print("=" * 50)
    print("СТАТИСТИКА ОБУЧЕНИЯ")
    print("=" * 50)
    print(f"Всего игр: {total_games}")
    print(f"")
    print(f"Результаты:")
    print(f"  Победы белых: {wins_white} ({wins_white/total_games*100:.1f}%)")
    print(f"  Победы чёрных: {wins_black} ({wins_black/total_games*100:.1f}%)")
    print(f"  Ничьи: {draws} ({draws/total_games*100:.1f}%)")
    print(f"")
    print(f"Длина игр (ходов):")
    print(f"  Среднее: {avg_moves:.1f}")
    print(f"  Минимум: {min_moves}")
    print(f"  Максимум: {max_moves}")
    print(f"  Среднее (последние 10): {avg_last_10:.1f}")
    print("=" * 50)
    
    # Тренд длины игр (первые 10 vs последние 10)
    if len(games) >= 20:
        first_10 = games[:10]
        avg_first_10 = sum(g['moves'] for g in first_10) / len(first_10)
        trend = avg_last_10 - avg_first_10
        trend_str = "↓" if trend < 0 else "↑" if trend > 0 else "→"
        print(f"Тренд: {avg_first_10:.0f} → {avg_last_10:.0f} ({trend_str} {abs(trend):.0f})")
        print("=" * 50)


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "distributed_training.log"
    parse_log(log_file)
