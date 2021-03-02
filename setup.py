from setuptools import setup, find_packages
import sys
import os.path


def load_requirements(filename='requirements.txt'):
    with open(filename) as f:
        lines = f.readlines()
    return lines


setup(
    name='aci',
    version="0.0.1",
    description='',
    url='',
    author='Levin Brinkmann',
    author_email='',
    license='',
    packages=[package for package in find_packages()
              if (package.startswith('aci'))],
    zip_safe=False,
    install_requires=load_requirements(),
    extras_require={'dev': load_requirements('dev_requirements.txt')},
    entry_points={
        'console_scripts': [
            'train = aci.train:main',
            'post = aci.post:main',
            'plot = aci.plot:main',
            'video = aci.video:main',
            'merge = aci.merge:main'
        ],
    },
    scripts=[
        'djx/scripts/djx'
    ]
)
