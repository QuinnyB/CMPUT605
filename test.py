from pynput import keyboard

def on_press(key):
    print(f"Key pressed: {key}")

# Start the Keyboard Listener Thread
listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    pass