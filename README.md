# ROSpider forked by Enigma Lab
###### Author: Connor Marsh
This repo is a fork of HIWONDER's repo, it introduces quality of life changes, extra documentation/usage instructions, and our additional code written for our researach done on the ROSpider robot.

### Quality of Life Changes
#### Prior usage
The robot had a systemd process that on startup runs the launch file bringup.launch.py, which runs all the controller files, sensor drivers, and the app service. This would consume a large amount of battery, mostly from the servos being enabled, and it was required to manually stop this service before running custom code
#### How to use now
We placed the systemd process with one that now runs the launch file bringup_core.launch.py, which runs just the necessary controller files and startup checks needed to operate the robot. We modified the controller_manager.py file to have "sleep" and "wake" services which put the robot into a resting pose and disable the servos, or put the robot into an active pose and enable the servos, respectively. The sleep service is called on startup of the controller_manager, so on robot's boot, the robot will move to the resting pose, and then disable the servos, saving energy.
###### rospider_run script
We have included a script `rospider_run` which is used as an execution wrapper around any code that desires to command the robots motors or sensors. If you want to run a command like `ros2 launch custom_autonomy autonomy_stack.launch.py`, you would execute it like this: `rospider_run ros2 launch custom_autonomy autonomy_stack.launch.py`

The rospider_run script will start by calling the controller_managers wake service, and then calling a launch file `bringup_sensors.launch.py` which starts the sensor suite, it then runs whatever command you put in, so in this case the `ros2 launch` command, and then it has a cleanup process which runs when this terminal is completed, so either on CTRL+C or when it completes naturally. The cleanup process searches for the process ID's of the sensor code, and kills those processes, and then it calls the sleep service of the controller_manager.

This system is highly convenient as it no longer requires you to start and stop the controller nodes everytime you want to run your code.