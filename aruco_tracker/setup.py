from setuptools import setup

package_name = 'aruco_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruth',
    maintainer_email='ruth@todo.todo',
    description='Aruco tracker node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'objects_state = aruco_tracker.aruco_tracker_node:main',
        ],
    },
)
