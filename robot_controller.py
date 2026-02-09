"""Main robot controller orchestrating all components."""
import time
import threading
import numpy as np
from pynput import keyboard

from config import RobotConfig
from robot_arm import RobotArm
from motion_controller import MotionController
from learning_agent import LearningAgent
from visualizer import Visualizer


class RobotController:
    """Main orchestrator for robot learning and control."""
    
    def __init__(self, config=None):
        """
        Initialize the robot controller.
        
        Args:
            config: RobotConfig object (uses defaults if None)
        """
        self.config = config or RobotConfig()
        
        # Initialize all components
        self.robot_arm = RobotArm(self.config)
        self.motion_controller = MotionController(self.robot_arm, self.config)
        self.learning_agent = LearningAgent(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        
        # State tracking
        self._running = False
        self.loop_count = 0
        self.avg_update_time = 0
        
        # Motor ID to read from
        self.motor_id = self.config.gripper.MOTOR_ID
    
    def _on_key_press(self, key):
        """Handle keyboard input for pause/resume."""
        if key == keyboard.Key.space:
            if self.motion_controller.is_paused():
                self.motion_controller.resume()
                print("\n*** RESUMED ***")
            else:
                self.motion_controller.pause()
                print("\n*** PAUSED ***")
    
    def start(self):
        """Start all background threads."""
        self.listener.start()
        self.motion_controller.start()
        self._running = True
        print("Robot Ready.")
        print("Starting live plot. Close the window to stop.")
        print("Press SPACE to pause/resume.")
    
    def run(self):
        """Main control loop."""
        self.start()
        
        try:
            # Main loop
            while self.visualizer.is_open() and self._running:
                start_time = time.perf_counter()
                
                # If paused, just flush events and wait
                if self.motion_controller.is_paused():
                    self.visualizer.flush_events()
                    time.sleep(0.1)
                    continue
                
                # Read motor state
                pos, vel, load = self.robot_arm.read_state(self.motor_id)
                
                if pos is not None:
                    # Perform learning step
                    step_data = self.learning_agent.step(pos, vel, load)
                    
                    # Calculate verifier (for retrospective TD validation)
                    verifier_value = None
                    if self.loop_count > self.learning_agent.verifier_buffer_length:
                        histories = self.visualizer.get_history()
                        cumulant_buffer = list(histories['cumulant'])[-self.learning_agent.verifier_buffer_length:]
                        verifier_value = np.sum(
                            np.array(cumulant_buffer) * 
                            (self.config.learning.GAMMA ** self.learning_agent.verifier_indices)
                        ) * (1 - self.config.learning.GAMMA)
                    
                    # Update visualization
                    self.visualizer.update(
                        pos=pos,
                        vel=vel,
                        load=load,
                        cumulant=step_data['c'],
                        prediction=step_data['pred'],
                        verifier_value=verifier_value
                    )
                
                # Update timing statistics
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                self.avg_update_time = self.avg_update_time + \
                    (elapsed_time - self.avg_update_time) / (self.loop_count + 1)
                self.loop_count += 1
                
                # Small delay to control update rate
                time.sleep(self.config.visualization.UPDATE_RATE)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean up and shut down all components."""
        print("\nShutting down...")
        self._running = False
        
        # Stop motion
        self.motion_controller.stop()
        
        # Stop keyboard listener
        if self.listener.is_alive():
            self.listener.stop()
        
        # Close visualization
        self.visualizer.close()
        
        # Close robot communication
        self.robot_arm.close()
        
        print("Shutdown complete.")
    
    def get_learning_stats(self):
        """
        Get current learning statistics.
        
        Returns:
            dict: Contains weights, feature vector, loop count, and timing
        """
        return {
            'weights': self.learning_agent.get_weights(),
            'feature_vector': self.learning_agent.get_feature_vector(),
            'loop_count': self.loop_count,
            'avg_update_time': self.avg_update_time
        }
    
    def print_stats(self):
        """Print learning statistics."""
        stats = self.get_learning_stats()
        print(f"\nAvg Loop Time: {stats['avg_update_time']:.4f} sec")
        print(f"Loop Count: {stats['loop_count']}")
        print(f"Weights: {stats['weights']}")


if __name__ == '__main__':
    controller = RobotController()
    controller.run()
