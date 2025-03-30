from setuptools import setup

package_name = 'arm_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 package for robotic arm reinforcement learning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = arm_rl.train:main',
            'test = arm_rl.test:main',
        ],
    },
)
