import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.animation import FuncAnimation
from tracker.tracker_main import Sort

# samples of detections for 10 frames
detections_per_frame = [
    [[100, 200, 150, 300]],
    [[105, 200, 155, 300]],
    [[110, 200, 160, 300]],
    [[115, 200, 165, 300]],
    [[120, 200, 170, 300]],
    [[125, 200, 175, 300]],
    [[130, 200, 180, 300]],
    [[135, 200, 185, 300]],
    [[140, 200, 190, 300]],
    [[145, 200, 195, 300]],
]

tracker = Sort()

fig, ax = plt.subplots()
ax.set_xlim(50, 250)
ax.set_ylim(350, 150)
ax.invert_yaxis()

det_patches = []
track_patches = []
track_texts = []

def init():
    ax.clear()
    ax.set_xlim(50, 250)
    ax.set_ylim(350, 150)
    ax.invert_yaxis()
    return []

def update(frame_idx):
    ax.clear()
    ax.set_xlim(50, 250)
    ax.set_ylim(0, 400)
    # ax.invert_yaxis()  # disattivata per test

    detections = detections_per_frame[frame_idx]
    tracked = tracker.update(detections)

    print(f"Frame {frame_idx} - Detections (red):")
    for bbox in detections:
        print(f"  bbox: {bbox}")

    for bbox in detections:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    print(f"Frame {frame_idx} - Tracked (green):")
    for obj_id, bbox in tracked:
        print(f"  ID: {obj_id}, bbox: {bbox}")

    for obj_id, bbox in tracked:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'ID: {obj_id}', color='g', fontsize=8)

    ax.set_title(f"Frame {frame_idx}")

    return []

ani = FuncAnimation(fig, update, frames=len(detections_per_frame), init_func=init, interval=500, repeat=False)

plt.show()
