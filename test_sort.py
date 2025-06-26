from tracker.tracker_main import Sort
import random
import matplotlib.pyplot as plt

tracker = Sort()

# Simula 10 frame con una persona che si muove
for frame in range(10):
    x = 100 + frame * 5 + random.uniform(-2, 2)
    y = 200 + random.uniform(-1, 1)
    bbox = [x, y, x+50, y+100]
    tracked = tracker.update([bbox])
    print(f"Frame {frame} → Tracked: {tracked}")
