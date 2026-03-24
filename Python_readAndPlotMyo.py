'''
Sample code for reading and plotting raw EMG data from a Myo armband. 
Required libraries: pyomyo, matplotlib
NOTE: You must not be connected to the Myo armband with any other software (e.g. Myo Connect) for this to work
    Right-click the Myo icon in your taskbar and select Quit or Exit Myo Connect before running this code.
Written by Google Gemini (with prompts from Quinn Boser)
March 2026
'''

import collections
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyomyo import Myo, emg_mode

# --- Configuration ---
WINDOW_SIZE = 100
NUM_CHANNELS = 8
data_buffers = [collections.deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(NUM_CHANNELS)]

def handle_emg(emg, worker):
    for i in range(NUM_CHANNELS):
        data_buffers[i].append(emg[i])

# --- Myo Worker Thread ---
def myo_worker(myo_device):
    print("Myo thread started...")
    while True:
        myo_device.run() # This loop now runs independently of the plot

# --- Plot Setup ---
fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Myo Raw EMG Channels (Real-time)', fontsize=16)
lines = [axes[i].plot(range(WINDOW_SIZE), [0]*WINDOW_SIZE, color='teal')[0] for i in range(NUM_CHANNELS)]

for i in range(NUM_CHANNELS):
    axes[i].set_ylim(-128, 127)
    axes[i].set_ylabel(f"Ch {i+1}")

def update_plot(frame):
    for i in range(NUM_CHANNELS):
        lines[i].set_ydata(list(data_buffers[i]))
    return lines

# --- Initialization ---
m = Myo(mode=emg_mode.RAW)
m.add_emg_handler(handle_emg)
m.connect()

# Start the background thread
t = threading.Thread(target=myo_worker, args=(m,), daemon=True)
t.start()

# Start the animation (this will "hold" the main thread)
ani = FuncAnimation(fig, update_plot, interval=20, blit=True, cache_frame_data=False)

try:
    print("Visualizer running. Close the plot window to stop.")
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    m.disconnect()