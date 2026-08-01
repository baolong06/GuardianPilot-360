import time
import random

class CANBridge:
    def __init__(self, interface="can0"):
        self.interface = interface
        # Trong thực tế: import can và setup
    
    def send_brake_signal(self, level):
        """Gửi tín hiệu AEB khi BRAKE triggered"""
        print(f"[CAN] Sending BRAKE signal: {level}")
        # Mô phỏng gửi CAN message
        # can_id = 0x123
        # data = [level]
        # can_bus.send(can_id, data)
    
    def send_warning(self, warning_level):
        if warning_level == "BRAKE":
            self.send_brake_signal(1)  # hard brake
        elif warning_level == "ALERT":
            self.send_brake_signal(0.5)  # partial brake
        else:
            pass