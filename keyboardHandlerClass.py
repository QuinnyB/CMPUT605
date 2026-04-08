from pynput import keyboard

class KeyPressHandler:
    def __init__(self, robot=None, motor_id=None):
        self.robot = robot
        self.motor_id = motor_id
        
        self.current_key = None  # For 'a', 's', 'd'
        self.is_paused = False   # For '`' toggle
        self.running = True
        self.space_pressed = False
        self.recalibrate_requested = False
        
        # Start the listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        # Hand spacebar press:
        if key == keyboard.Key.space:
            self.space_pressed = True
            return  
        try:
            k = key.char
            # ` Key to Pause/Resume
            if k == '`':
                self.is_paused = not self.is_paused
                print(f"*** {'PAUSED' if self.is_paused else 'RESUMED'} ***")
                if self.is_paused and self.robot is not None and self.motor_id is not None:
                    # Emergency stop: read current pos and set as goal
                    p, _, _ = self.robot.read_from_motor(self.motor_id)
                    self.robot.set_goal_pos(self.motor_id, p)
            # 'r' key to recalibrate IMU control
            if k == 'r':
                self.recalibrate_requested = True 
            # a/s/d keys to indicate intended action for learner
            elif k in ['a', 's', 'd']:
                self.current_key = k       
        except AttributeError:
            # Handles special keys (like Shift/Ctrl) that don't have .char
            pass

    def on_release(self, key):
        try:
            # Only clear the action if the key released is the one currently active
            if hasattr(key, 'char') and key.char == self.current_key:
                self.current_key = None
        except AttributeError:
            pass

    def get_key(self):
        return self.current_key

    def get_paused(self):
        return self.is_paused
    
    def reset_space(self):
        self.space_pressed = False
    
    def stop(self):
        self.running = False
        self.listener.stop()
