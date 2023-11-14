from setuptools import find_packages, setup

setup(
    name='AlgoWorld',
    packages=find_packages(include=['AlgoWorld']),
    version='0.1.0',
    description='A Python Library aimed to solve various problems using well known algorithms',
    author='Tymon Dydowicz',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)