#!/usr/bin/env python3
# encoding: utf-8

import os
import psutil
import threading
import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import BuzzerState, OLEDState

def check_mic():
    data = os.popen('ls /dev/ |grep ring_mic').read()
    if data == 'ring_mic\n':
        os.system("ros2 launch xf_mic_asr_offline startup_test.launch.py")

def get_cpu_serial_number():
    try:
        device_serial_number = open("/proc/device-tree/serial-number")
        serial_num = device_serial_number.readlines()[0][-10:-1]
        sn = (serial_num + "00000000000000000000000000")[:32]
        return ''.join(["WN-", sn[0:8]])
    except Exception:
        return "WN-UNKNOWN"

class StartupCheckNode(Node):
    def __init__(self):
        super().__init__('startup')
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1)
        self.oled_pub = self.create_publisher(OLEDState, '/ros_robot_controller/set_oled', 1)
        
        # Trigger the initial buzzer sequence once on boot
        self.play_buzzer()
        
        # Start a timer to check the IP and update the OLED every 5 seconds
        self.timer = self.create_timer(10.0, self.update_oled)

    def play_buzzer(self):
        msg = BuzzerState()
        msg.freq = 1900
        msg.on_time = 0.2
        msg.off_time = 0.01
        msg.repeat = 1
        self.buzzer_pub.publish(msg)

    def get_wlan_ip(self):
        # Wrap the psutil call in a try/except to gracefully handle temporary network dropouts
        try:
            info = psutil.net_if_addrs()
            for k, v in info.items():
                if 'wl' in k:  # Matches wlan0
                    for i in v:
                        # AF_INET is 2 (IPv4). This safely grabs the standard IPv4 address.
                        if i.family == 2:
                            return i.address
        except Exception as e:
            self.get_logger().warn(f'Error reading network interfaces: {e}')
        
        return '0.0.0.0'

    def update_oled(self):
        # Publish SSID line
        msg_ssid = OLEDState()
        msg_ssid.index = 1
        msg_ssid.text = 'SSID:' + get_cpu_serial_number()
        self.oled_pub.publish(msg_ssid)
        
        # Publish dynamic IP line
        msg_ip = OLEDState()
        msg_ip.index = 2
        msg_ip.text = 'IP:' + self.get_wlan_ip()
        self.oled_pub.publish(msg_ip)

def main(args=None):
    # Keep the microphone check threaded (and set to daemon) so it doesn't block the node
    threading.Thread(target=check_mic, daemon=True).start()
    
    rclpy.init(args=args)
    node = StartupCheckNode()
    
    try:
        # Spin keeps the node alive indefinitely to run the timer callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
