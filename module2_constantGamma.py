"""
Main script for robot learning with temporal difference learning.

This script demonstrates a complete learning system using object-oriented design
with separate concerns for hardware control, motion planning, learning algorithm,
and visualization.

To use:
    python module2_constantGamma.py
    Press SPACE to pause/resume motion
    Close the plot window to exit
"""

from robot_controller import RobotController
from config import RobotConfig


def main():
    """Main entry point for the learning system."""
    # Create configuration (can be modified here if needed)
    config = RobotConfig()
    
    # Uncomment to customize settings:
    # config.hardware.COMM_PORT = 'COM14'
    # config.learning.GAMMA = 0.7
    # config.learning.ALPHA = 0.5
    
    # Create and run controller
    controller = RobotController(config)
    controller.run()


if __name__ == '__main__':
    main()