with open('game_dataset.csv', 'r', encoding='utf-8', errors='ignore') as f:
    for i in range(5):
        print(f"Line {i}: {f.readline().strip()}")
