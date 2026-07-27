#关闭所有ros节点
~/.stop_ros.sh

#编译
cd ~/ros2_ws 
colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install
#单独编译某个包
colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install --packages-select xxx

#深度摄像头点云可视化
#深度摄像头RGB图像可视化
ros2 launch peripherals depth_camera.launch.py
rviz2

#雷达数据可视化
ros2 launch peripherals lidar_view.launch.py

#imu校准
ros2 launch ros_robot_controller ros_robot_controller.launch.py
ros2 run imu_calib do_calib --ros-args -r imu:=/ros_robot_controller/imu_raw --param output_file:=/home/ubuntu/ros2_ws/src/peripherals/config/imu_calib.yaml

#查看imu校准效果
ros2 launch peripherals imu_view.launch.py

#######app#######
#姿态自平衡
ros2 launch app self_balancing_node.launch.py debug:=true

#雷达功能
ros2 launch app lidar_node.launch.py debug:=true
#雷达避障
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 1}"

#雷达跟随
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 2}"

#雷达警卫
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 3}"

#追踪踢球
ros2 launch app intelligent_kick_node.launch.py debug:=true
ros2 service call /intelligent_kick/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /intelligent_kick/set_running std_srvs/srv/SetBool "{data: True}"

#目标追踪
ros2 launch app object_tracking_node.launch.py debug:=true
ros2 service call /object_tracking/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /object_tracking/set_running std_srvs/srv/SetBool "{data: True}"


#巡线
ros2 launch app line_following_node.launch.py debug:=true
ros2 service call /line_following/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"

#手势控制
ros2 launch app hand_gesture.launch.py debug:=true
ros2 service call /hand_gesture/enter std_srvs/srv/Trigger {}
ros2 service call /hand_gesture/set_running std_srvs/srv/SetBool "{data: True}"

#######example#######
#######ROS六足机器人控制基础课程######

#机体直行与转弯
ros2 launch example forward_and_rorate.launch.py

#左右平移
ros2 launch example left_and_right.launch.py

#斜向平移
ros2 launch example diagonally.launch.py

#行走速度调节
ros2 launch example speed_control.launch.py

#折线平移
ros2 launch example broken_line_walk.launch.py

#矩形平移
ros2 launch example square_walk.launch.py

#OLED液晶屏显示
ros2 launch example oled.launch.py

######ROS六足机器人逆运动学控制######

#六足逆运动学控制
ros2 launch example body_ik.launch.py

#行走高度调节
ros2 launch example height_adjustment.launch.py

#躯干姿态调节
ros2 launch example posture_adjustment.launch.py

#机体扭动实验
ros2 launch example body_wave.launch.py

#机体舞动实验（质心）
ros2 launch example body_circle.launch.py

#ROS机器人姿态自平衡
ros2 launch app self_balancing_node.launch.py debug:=true

#######slam#######
#2D建图
ros2 launch slam slam.launch.py

#rviz查看建图效果
ros2 launch slam rviz_slam.launch.py

#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py

#保存地图
#/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml
cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true

#cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "保存名称" --ros-args -p map_subscribe_transient_local:=true -r __ns:=/robot_1

#3D建图，仅支持深度相机版本
ros2 launch slam rtabmap_slam.launch.py

#rviz查看建图效果
ros2 launch slam rviz_rtabmap.launch.py

#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py

#######navigation#######
#2D导航
##rviz发布导航目标
ros2 launch navigation rviz_navigation.launch.py
ros2 launch navigation navigation.launch.py map:=地图名称

#3D导航
ros2 launch navigation rtabmap_navigation.launch.py

#rviz发布导航目标
ros2 launch navigation rviz_rtabmap_navigation.launch.py

#######opencv#######
#颜色识别
ros2 launch example color_recognition_node.launch.py

#AprilTag标签识别
ros2 launch example apriltag_recognition.launch.py

#AR视觉
ros2 launch example ar.launch.py

#色块坐标定位
ros2 launch example color_position.launch.py

#颜色追踪
ros2 launch app object_tracking_node.launch.py debug:=true
ros2 service call /object_tracking/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /object_tracking/set_running std_srvs/srv/SetBool "{data: True}"

#AprilTag标签定位
ros2 launch example apriltag_position.launch.py

#AprilTag标签追踪
ros2 launch example apriltag_track.launch.py

#巡线行驶
ros2 launch app line_following_node.launch.py debug:=true
ros2 service call /line_following/enter std_srvs/srv/Trigger {}

#鼠标左键点击画面取色
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"

#KCF目标识别
ros2 launch example kcf.launch.py
######Yolov8######

#yolo目标识别
ros2 launch example yolov8_object_detection_node.launch.py yolo:="yolov8"/"yolov11"

#垃圾分类
ros2 launch example garbage_classification.launch.py yolo:="yolov8"/"yolov11"

######MediaPipe######
#指尖轨迹识别
ros2 launch example finger_trajectory.launch.py

#手势识别
ros2 launch example hand_detect.launch.py

#手势控制
ros2 launch example hand_gesture.launch.py

#人脸检测与追踪
ros2 launch example face_track.launch.py

#肢体体感控制
ros2 launch example pose_control.launch.py

######rgbd ，仅支持深度相机版本######
#三维物体抓取
ros2 launch example track_and_grab.launch.py

#防跌落 
ros2 launch example prevent_falling.launch.py

#过独木桥
ros2 launch example cross_bridge.launch.py

#机械臂追踪
ros2 launch example color_track_node.launch.py

#追踪踢球
ros2 launch example intelligent_kick.launch.py

#三维形状识别夹取
ros2 launch example object_classification.launch.py

#体积测量
ros2 launch example object_volume_measurement.launch.py
##############导航搬运#############

#夹取校准
ros2 launch example automatic_pick.launch.py debug:=pick
#放置校准
ros2 launch example automatic_pick.launch.py debug:=place
#开启夹取
ros2 service call /automatic_pick/pick std_srvs/srv/Trigger {}
#开启放置
ros2 service call /automatic_pick/place std_srvs/srv/Trigger {}

#导航搬运
ros2 launch example navigation_transport.launch.py map:=map_01

#######xf_mic_asr_offline#######

#语音控制移动
ros2 launch xf_mic_asr_offline voice_control_move.launch.py

#######simulations#######
#urdf可视化
ros2 launch rospider_description display.launch.py

########software######

#lab_tool
python3 ~/software/lab_tool/main.py

#collect_picture
python3 ~/software/collect_picture/main.py

#servo_tool
python3 ~/software/servo_tool/main.py

#actionset_editor
python3 ~/software/actionset_editor/main.py

#######large_models_examples--online#######
#需要使用Tool工具将语音模式切换到对应模式
#ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "前进后退左转然后右转"}'
#前后左右，横向移动
ros2 launch large_models_examples llm_control_move.launch.py

#追踪红色物体
ros2 launch large_models_examples llm_color_track.launch.py

#沿着黑线走，遇到障碍就停下
ros2 launch large_models_examples llm_visual_patrol.launch.py

#描述下你看到了什么
ros2 launch large_models_examples vllm_with_camera.launch.py

#跟踪前面小车
ros2 launch large_models_examples vllm_track.launch.py

#去动物园看看有哪些动物，然后回来告诉我
ros2 launch large_models_examples vllm_navigation.launch.py map:=map_01

#夹取/放置校准
ros2 launch large_models_examples automatic_pick.launch.py debug:=pick/debug:=place

#将红色方块放到对应颜色的盒子里
ros2 launch large_models_examples vllm_navigation_transport.launch.py map:=map_01

#function calling

ros2 launch large_models_examples llm_control_progress.launch.py

ros2 launch large_models_examples llm_control_progress.launch.py function:=navigation

#######large_models_examples--offline#######
#需要使用Tool工具将语音模式切换到对应模式
#ros2 topic pub --once /vocal_detect/asr_result std_msgs/msg/String '{data: "前进后退左转然后右转"}'
#前后左右，横向移动
ros2 launch large_models_examples llm_control_move.launch.py

#追踪红色物体
ros2 launch large_models_examples llm_color_track.launch.py

#沿着黑线走，遇到障碍就停下
ros2 launch large_models_examples llm_visual_patrol.launch.py

################# competition ###############
#赛事启动程序
ros2 launch competition competition.launch.py 

# 过窄通道
ros2 service call /narrow_slit_traversal/enter std_srvs/srv/Trigger {}
# 过独木桥
ros2 service call /cross_bridge/enter std_srvs/srv/Trigger {}
# 开启夹取
ros2 service call /automatic_pick/pick std_srvs/srv/Trigger {} 
# 开启放置
ros2 service call /automatic_pick/place std_srvs/srv/Trigger {} 
# 开启上台阶
ros2 service call /automatic_pick/is_up_steps std_srvs/srv/Trigger {}

