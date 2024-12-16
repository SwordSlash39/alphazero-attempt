import time
from Trainer import Trainer

# Load model
ml = Trainer()
try:
    ml.loadmodel()
except FileNotFoundError:
    print("No existing model found, training new model")
    
def read_positions(file):
    positions = []
    with open(file, 'r') as file:
        for line in file:
            positions.append(line.strip())
    
    return positions

# Rest is up to me
s = time.time()
games = 0
save_count = 10

try:
    while True:
        t = time.time()
        ml.train(10, n=10000)
        games += 10
        print(f"Game {games}; Time taken: {time.time() - t:.2f}s")
        ml.savemodel()
except KeyboardInterrupt:
    pass

print(f"\n{games} games played, avg time per game: {(time.time() - s)/games:.2f}s")
# ml.savemodel()

""" for running on itself
s = time.time()
games = 1
save_count = 10

try:
    while True:
        t = time.time()
        ml.train(100)
        games += 100
        print(f"Game {games}; Time taken: {time.time() - t:.2f}s")
except KeyboardInterrupt:
    pass

print(f"\n{games} games played, avg time per game: {(time.time() - s)/games:.2f}s")
"""

"""
try:
    while True:
        cycle = read_positions("positions.txt")
        random.shuffle(cycle)
        for p in cycle:
            t = time.time()
            ml.train(1, maxmoves=4, custom_fen=p)
            games += 1
            print(f"Game {games}; Time taken: {time.time() - t:.2f}s")
except KeyboardInterrupt:
    pass

print(f"\n{games} games played, avg time per game: {(time.time() - s)/games:.2f}s")
"""