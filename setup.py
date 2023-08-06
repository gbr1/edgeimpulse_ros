from setuptools import setup

package_name = 'edgeimpulse_ros'
submodules = 'edgeimpulse_ros/submodules'
setup(
    name=package_name,
    version='0.0.5',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giovanni di Dio Bruno',
    maintainer_email='giovannididio.bruno@gmail.com',
    description='ROS2 wrapper for Edge Impulse',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_classification = edgeimpulse_ros.image_classification:main'
        ],
    },
)
