"""Robot arm hardware abstraction and communication."""
from dynamixel_sdk import PortHandler, PacketHandler
from robotModuleFunctions import to_signed_32, to_signed_16


class RobotArm:
    """Handles communication with Dynamixel motors."""
    
    def __init__(self, config):
        """
        Initialize robot arm and establish motor communication.
        
        Args:
            config: RobotConfig object containing hardware settings
        """
        self.config = config
        self.hw_cfg = config.hardware
        self.motor_addr = config.motor_addresses
        
        # Initialize communication
        self.port_handler = PortHandler(self.hw_cfg.COMM_PORT)
        self.packet_handler = PacketHandler(self.hw_cfg.PROTOCOL_VERSION)
        
        # Open port and set baudrate
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.hw_cfg.COMM_PORT}")
        if not self.port_handler.setBaudRate(self.hw_cfg.BAUDRATE):
            raise RuntimeError(f"Failed to set baudrate to {self.hw_cfg.BAUDRATE}")
        
        # Initialize all motors
        self._initialize_motors()
    
    def _initialize_motors(self):
        """Enable torque and set initial parameters for all motors."""
        for motor_id in self.hw_cfg.MOTOR_IDS:
            # Enable torque
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, 
                self.motor_addr.TORQUE_ENABLE, 1
            )
            # Set profile velocity
            self.packet_handler.write4ByteTxRx(
                self.port_handler, motor_id,
                self.motor_addr.PROFILE_VELOCITY, self.config.gripper.MOTOR_VELOCITY
            )
            # Move to initial position
            initial_pos = self.hw_cfg.INITIAL_POSITIONS[motor_id]
            self.packet_handler.write4ByteTxRx(
                self.port_handler, motor_id,
                self.motor_addr.GOAL_POSITION, initial_pos
            )
    
    def read_state(self, motor_id):
        """
        Read position, velocity, and load from a specific motor.
        
        Args:
            motor_id: ID of the motor to read from
            
        Returns:
            tuple: (position, velocity, load) or (None, None, None) on error
        """
        pos, _, _ = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.PRESENT_POSITION
        )
        
        if pos is None:
            return None, None, None
        
        vel, _, _ = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.PRESENT_VELOCITY
        )
        
        load, _, _ = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.PRESENT_LOAD
        )
        
        # Convert to signed values
        vel_signed = to_signed_32(vel) if vel is not None else 0
        load_signed = to_signed_16(load) if load is not None else 0
        
        return pos, vel_signed, load_signed
    
    def set_goal_position(self, motor_id, position):
        """
        Set the goal position for a motor.
        
        Args:
            motor_id: ID of the motor
            position: Target position (0-4095)
        """
        self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.GOAL_POSITION, position
        )
    
    def set_profile_velocity(self, motor_id, velocity):
        """
        Set the profile velocity for a motor.
        
        Args:
            motor_id: ID of the motor
            velocity: Velocity value
        """
        self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.PROFILE_VELOCITY, velocity
        )
    
    def enable_torque(self, motor_id):
        """Enable torque for a motor."""
        self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.TORQUE_ENABLE, 1
        )
    
    def disable_torque(self, motor_id):
        """Disable torque for a motor."""
        self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id,
            self.motor_addr.TORQUE_ENABLE, 0
        )
    
    def close(self):
        """Close the communication port and disable all motors."""
        for motor_id in self.hw_cfg.MOTOR_IDS:
            self.disable_torque(motor_id)
        self.port_handler.closePort()
