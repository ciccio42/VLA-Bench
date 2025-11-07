from distutils.core import setup
from setuptools import find_packages

setup(
    name='test',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    license='MIT License'
)

if __name__ == '__main__':
    print(f"Find packages: {find_packages()}")