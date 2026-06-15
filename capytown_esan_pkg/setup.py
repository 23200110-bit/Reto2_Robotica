from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'capytown_esan_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ESAN_Alumnos',
    maintainer_email='alumnos@esan.edu.pe',
    description='Paquete de seguimiento de carril autónomo - Reto 2 (CapyTown)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = capytown_esan_pkg.lane_detector:main',
            'lane_controller = capytown_esan_pkg.lane_controller:main',
        ],
    },
)
