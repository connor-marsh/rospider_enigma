cmake_minimum_required(VERSION 3.22)
project(robot_moveit_config)

find_package(ament_cmake REQUIRED)

# MIGRATED: install() calls moved ABOVE ament_package().
# ament_package() must be the last command in the file.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

ament_package()
