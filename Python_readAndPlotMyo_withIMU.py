import threading
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyomyo import Myo, emg_mode
from helperFunctions import *

# --- Shared Data ---
latest_data = {
    "raw_matrix": np.eye(3),
    "cal_matrix": np.eye(3),
    "is_calibrated": False,
    "emg": [0] * 8
}
data_lock = threading.Lock()

# --- Handlers ---
def handle_emg(emg, worker):
    with data_lock:
        latest_data["emg"] = list(emg)

def handle_imu(quat, acc, gyro):
    R = quat_to_mat(quat[0], quat[1], quat[2], quat[3])
    with data_lock:
        latest_data["raw_matrix"] = R

def on_press(event):
    if event.key == ' ':
        with data_lock:
            latest_data["cal_matrix"] = np.copy(latest_data["raw_matrix"])
            latest_data["is_calibrated"] = True
        print("\n[CALIBRATED]")

# --- Plot Setup ---
fig = plt.figure(figsize=(12, 6))
fig.canvas.mpl_connect('key_press_event', on_press)
# Split screen: Left for EMG, Right for Matrix
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

# Left: EMG Bars
ax_emg = fig.add_subplot(gs[0, 0])
emg_bars = ax_emg.bar(range(8), [0]*8, color='teal')
ax_emg.set_ylim(-128, 128)
ax_emg.set_title("Raw EMG Levels")
ax_emg.set_xticks(range(8))
ax_emg.set_xticklabels([f"C{i+1}" for i in range(8)])

# Right: Matrix Text
ax_mat = fig.add_subplot(gs[0, 1])
ax_mat.axis('off')
ax_mat.set_title("Rotation Matrix (Space to Calibrate)")

mat_texts = []
for i in range(3):
    for j in range(3):
        # Adjusted spacing to prevent overlapping
        t = ax_mat.text(j*0.3 + 0.2, 0.7 - i*0.2, "0.00", 
                        ha='center', va='center', fontsize=14, 
                        fontweight='bold', family='monospace')
        mat_texts.append(t)

angle_texts = []
for i, angle_name in enumerate(['Z', 'X', 'Y']):
    t = ax_mat.text(0.5, 0.1 - i*0.05, f"{angle_name}: 0.00°", 
                    ha='center', va='center', fontsize=12, 
                    fontweight='bold', family='monospace')
    angle_texts.append(t)

status_text = ax_mat.text(0.5, 0.9, "RAW MODE", ha='center', color='blue', fontsize=10)

def update_plot(frame):
    with data_lock:
        R_now = latest_data["raw_matrix"]
        R_cal = latest_data["cal_matrix"]
        emg = latest_data["emg"]
        calibrated = latest_data["is_calibrated"]

    # Update EMG Bars
    for i, bar in enumerate(emg_bars):
        bar.set_height(abs(emg[i]))

    # Calculate and Update Matrix
    R_display = R_cal.T @ R_now if calibrated else R_now
    angle_display = mat_to_ZXY(R_display)
    status_text.set_text("CALIBRATED (RELATIVE)" if calibrated else "RAW MODE")
    status_text.set_color("green" if calibrated else "blue")

    flat_matrix = R_display.flatten()
    for i, val in enumerate(flat_matrix):
        mat_texts[i].set_text(f"{val:+.2f}") # 2 decimals to save space
        mat_texts[i].set_color("forestgreen" if val >= 0 else "crimson")
    
    for i, angle in enumerate(angle_display):
        angle_texts[i].set_text(f"{['Z', 'X', 'Y'][i]}: {angle:+.1f}°")
        angle_texts[i].set_color("forestgreen" if abs(angle) < 45 else "crimson")

    return list(emg_bars) + mat_texts + angle_texts + [status_text]

# --- Myo Thread with Crash Protection ---
def myo_worker(myo_device):
    while True:
        try:
            myo_device.run()
        except Exception:
            continue

m = Myo(mode=emg_mode.RAW)
m.add_emg_handler(handle_emg)
m.add_imu_handler(handle_imu)
m.connect()

threading.Thread(target=myo_worker, args=(m,), daemon=True).start()

ani = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
plt.show()
m.disconnect()