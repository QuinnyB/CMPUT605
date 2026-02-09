"""Gripper motion control and background movement logic."""
import threading
import time


class MotionController:
    """Controls repetitive gripper motion (open/close cycling)."""
    
    def __init__(self, robot_arm, config):
        """
        Initialize motion controller.
        
        Args:
            robot_arm: RobotArm instance for motor control
            config: RobotConfig object containing gripper settings
        """
        self.robot_arm = robot_arm
        self.config = config
        self.gripper_cfg = config.gripper
        
        self._running = False
        self._paused = False
        self._motion_thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background gripper motion thread."""
        if self._running:
            return
        
        self._running = True
        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()
    
    def stop(self):
        """Stop the background motion thread and clean up."""
        self._running = False
        if self._motion_thread:
            self._motion_thread.join(timeout=2.0)
    
    def pause(self):
        """Pause gripper motion and lock it in place."""
        with self._lock:
            if not self._paused:
                self._paused = True
                # Read current position and set it as goal to stop immediately
                pos, _, _ = self.robot_arm.read_state(self.gripper_cfg.MOTOR_ID)
                if pos is not None:
                    self.robot_arm.set_goal_position(self.gripper_cfg.MOTOR_ID, pos)
    
    def resume(self):
        """Resume gripper motion."""
        with self._lock:
            self._paused = False
    
    def is_paused(self):
        """Check if motion is paused."""
        with self._lock:
            return self._paused
    
    def _motion_loop(self):
        """Background thread routine for cyclic gripper motion."""
        while self._running:
            # Check for pause
            with self._lock:
                if self._paused:
                    pass  # Stay paused
                else:
                    # Move to closed position
                    self.robot_arm.set_goal_position(
                        self.gripper_cfg.MOTOR_ID,
                        self.gripper_cfg.POSITION_CLOSED
                    )
                    time.sleep(self.gripper_cfg.wait_time)
                    
                    # Move to open position
                    self.robot_arm.set_goal_position(
                        self.gripper_cfg.MOTOR_ID,
                        self.gripper_cfg.POSITION_OPEN
                    )
                    time.sleep(self.gripper_cfg.wait_time)
            
            # Small sleep to prevent busy waiting while paused
            if self.is_paused():
                time.sleep(0.1)
