from setuptools import find_packages, setup

setup(
    name='gui_controller',
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/gui_controller']),
        ('share/gui_controller', ['package.xml']),
        ('lib/gui_controller', ['scripts/gui_controller_node']),
    ],
)
