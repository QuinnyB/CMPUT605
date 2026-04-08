import time
from robotClass import MiniBento
from myoClass import MyoArmband
from learnerClass import ACLearner_Discrete
from visualizerClass import ACVisualizer
from keyboardHandlerClass import KeyPressHandler
from helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Myo Armband:
EMG_MAV_WINDOW = 40        # Window size for moving average of EMG signals (in number of samples)
IMU_MAV_WINDOW = 10        # Window size for moving average of IMU signals (in number of samples)
calibration_steps = [
    ("Internally rotate shoulder to limit", "int_rot", 0),
    ("Externally rotate shoulder to limit", "ext_rot", 0),
    ("Flex your elbow to limit",           "flex",    2),
    ("Extend your elbow to limit",          "ext",     2)
]
imu_cal_angles = {
    "int_rot": 0,
    "ext_rot": 0,
    "flex": 0,
    "ext": 0   
}

# Robot arm:
COMM_PORT = 'COM13'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
SHO_ID = 1
SHO_LIMS = [1500, 2500]
ELB_ID = 2
ELB_LIMS = [1300, 2700]
HAND_ID = 5  
HAND_LIMS = [1750, 2650] 

with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm, MyoArmband(EMG_MAV_WINDOW, IMU_MAV_WINDOW) as myo:
# with MyoArmband(EMG_MAV_WINDOW, IMU_MAV_WINDOW) as myo:
    # key_handler = KeyPressHandler(arm, MOTOR_ID)
    key_handler = KeyPressHandler()

    # Calibrate IMU in neutral arm position
    print("\n" + "="*50)
    print("IMU CALIBRATION STEP 1")
    print("Hold your arm in the 'Neutral' position. Press SPACEBAR to calibrate.")
    print("="*50 + "\n")
    while not key_handler.space_pressed: time.sleep(0.1)
    myo.calibrate_imu()
    key_handler.reset_space()

    # Get angle limits for internal/external rotation and flexion/extension
    for i, (msg, key, idx) in enumerate(calibration_steps, start=2):
        print(f"\n" + "="*50 + f"\nSTEP {i}: {msg}\nHold position and press SPACE\n" + "="*50)
        while not key_handler.space_pressed: time.sleep(0.1)
        _, angles, _, _, _ = myo.get_data() 
        imu_cal_angles[key] = angles[idx]
        print(f"Recorded {key}: {imu_cal_angles[key]:.2f}°")
        key_handler.reset_space()  

    # Check if we need to invert shoulder or elbow angles based on calibration     
    if imu_cal_angles["ext_rot"] < imu_cal_angles["int_rot"]:
        sho_mult = -1
        imu_cal_angles["int_rot"], imu_cal_angles["ext_rot"] = -imu_cal_angles["int_rot"], -imu_cal_angles["ext_rot"]
    else:
        sho_mult = 1
    if imu_cal_angles["flex"] < imu_cal_angles["ext"]:
        elb_mult = -1
        imu_cal_angles["flex"], imu_cal_angles["ext"] = -imu_cal_angles["flex"], -imu_cal_angles["ext"]
    else:
        elb_mult = 1

    while True:
        if not key_handler.get_paused():
            _, angles, _, _, _ = myo.get_data()
            print(f"Angle MAV: {angles}")
        
            # Set Mini Bento shoulder and elbow position based on IMU angles
            target_s = np.interp(sho_mult * angles[0], 
                             [imu_cal_angles["int_rot"], imu_cal_angles["ext_rot"]], 
                             SHO_LIMS)
            target_e = np.interp(elb_mult * angles[2], 
                             [imu_cal_angles["ext"], imu_cal_angles["flex"]], 
                             ELB_LIMS)
            
            print(f"Target SHO: {target_s:.0f}, Target ELB: {target_e:.0f}")

            arm.set_goal_pos(1, int(target_s))
            arm.set_goal_pos(2, int(target_e))


        time.sleep(0.1)