from setuptools import find_packages, setup

package_name = 'safeguard_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zinzi',
    maintainer_email='zinzi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'detection_publisher = safeguard_detection.detection_publisher:main','mission_planner = safeguard_detection.mission_planner:main',
        ],
    },
    
)
