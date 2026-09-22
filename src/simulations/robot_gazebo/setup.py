import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.*'))),
        (os.path.join('share', package_name, 'maps'), glob(os.path.join('maps', '*.*'))),
        (os.path.join('share', package_name, 'launch/include'), glob(os.path.join('launch/include', '*.*'))),

        
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*.*'))),
        (os.path.join('share', package_name, 'meshes/bed'), glob(os.path.join('meshes/bed', '*.*'))),
        (os.path.join('share', package_name, 'meshes/bookshelft'), glob(os.path.join('meshes/bookshelft', '*.*'))),
        (os.path.join('share', package_name, 'meshes/chair'), glob(os.path.join('meshes/chair', '*.*'))),
        (os.path.join('share', package_name, 'meshes/cupboard'), glob(os.path.join('meshes/cupboard', '*.*'))),
        (os.path.join('share', package_name, 'meshes/sofa'), glob(os.path.join('meshes/sofa', '*.*'))),
        (os.path.join('share', package_name, 'meshes/table'), glob(os.path.join('meshes/table', '*.*'))),
        (os.path.join('share', package_name, 'meshes/tea_table'), glob(os.path.join('meshes/tea_table', '*.*'))),
        (os.path.join('share', package_name, 'meshes/wpr1'), glob(os.path.join('meshes/wpr1', '*.*'))),
        
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.*'))),
        (os.path.join('share', package_name, 'models/bed'), glob(os.path.join('models/bed', '*.*'))),
        (os.path.join('share', package_name, 'models/bottles'), glob(os.path.join('models/bottles', '*.*'))),
        (os.path.join('share', package_name, 'models/cupboard'), glob(os.path.join('models/cupboard', '*.*'))),
        (os.path.join('share', package_name, 'models/table'), glob(os.path.join('models/table', '*.*'))),
        (os.path.join('share', package_name, 'models/chair'), glob(os.path.join('models/chair', '*.*'))),
        (os.path.join('share', package_name, 'meshes/kai'), glob(os.path.join('meshes/kai', '*.*'))),
        (os.path.join('share', package_name, 'meshes/man_1'), glob(os.path.join('meshes/man_1', '*.*'))),
        (os.path.join('share', package_name, 'meshes/man_2'), glob(os.path.join('meshes/man_2', '*.*'))),
        (os.path.join('share', package_name, 'meshes/woman'), glob(os.path.join('meshes/woman', '*.*'))),
        (os.path.join('share', package_name, 'meshes/wpb_home'), glob(os.path.join('meshes/wpb_home', '*.*'))),
        (os.path.join('share', package_name, 'meshes/wpv3'), glob(os.path.join('meshes/wpv3', '*.*'))),
        (os.path.join('share', package_name, 'models/balls'), glob(os.path.join('models/balls', '*.*'))),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # MIGRATED: robot_gazebo/teleop_key_control.py is MISSING from the source
            # tree (only a stale .pyc existed). Restore the file or delete this line.
            # 'teleop_key_control = robot_gazebo.teleop_key_control:main',
        ],
    },
)
