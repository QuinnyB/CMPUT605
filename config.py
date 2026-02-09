"""Configuration classes for the robot learning system."""
from dataclasses import dataclass
import math


@dataclass
class MotorAddresses:
    """Register addresses for Dynamixel motor communication."""
    TORQUE_ENABLE: int = 64
    GOAL_POSITION: int = 116
    PROFILE_VELOCITY: int = 112
    PRESENT_LOAD: int = 126
    PRESENT_VELOCITY: int = 128
    PRESENT_POSITION: int = 132


@dataclass
class HardwareConfig:
    """Hardware communication and motor configuration."""
    COMM_PORT: str = 'COM13'
    BAUDRATE: int = 1000000
    PROTOCOL_VERSION: float = 2.0
    MOTOR_IDS: list = None
    INITIAL_POSITIONS: dict = None
    
    def __post_init__(self):
        if self.MOTOR_IDS is None:
            self.MOTOR_IDS = [1, 2, 4, 5]
        if self.INITIAL_POSITIONS is None:
            self.INITIAL_POSITIONS = {1: 2048, 2: 1980, 4: 2048, 5: 2780}


@dataclass
class GripperConfig:
    """Gripper movement configuration."""
    MOTOR_ID: int = 5
    MOTOR_VELOCITY: int = 20
    POSITION_CLOSED: int = 1750
    POSITION_OPEN: int = 2650
    
    @property
    def position_range(self) -> int:
        return self.POSITION_OPEN - self.POSITION_CLOSED
    
    @property
    def wait_time(self) -> float:
        """Wait time in seconds for a full gripper cycle."""
        return math.ceil((self.position_range / (4096 * 0.229 * self.MOTOR_VELOCITY)) * 60)


@dataclass
class LearningConfig:
    """Temporal difference learning parameters."""
    MAX_LOAD: int = 300           # Max expected absolute load for normalization
    LOAD_THRESHOLD: int = 100     # Load threshold for cumulant signal
    NUM_POS_BINS: int = 10        # Position feature bins
    NUM_VEL_BINS: int = 20        # Velocity feature bins
    GAMMA: float = 0.5            # Discount factor
    ALPHA: float = 1.0            # Learning rate


@dataclass
class VisualizationConfig:
    """Real-time plotting configuration."""
    WINDOW_SIZE: int = 200        # Number of points to display
    UPDATE_RATE: float = 0.01     # seconds between plot updates
    FIGSIZE: tuple = (10, 6)      # Figure size


class RobotConfig:
    """Main configuration class that aggregates all settings."""
    
    def __init__(self):
        self.hardware = HardwareConfig()
        self.gripper = GripperConfig()
        self.learning = LearningConfig()
        self.visualization = VisualizationConfig()
        self.motor_addresses = MotorAddresses()
