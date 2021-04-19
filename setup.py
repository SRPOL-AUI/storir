"""Library setup."""
import setuptools
from setuptools import setup

setup(
    name='storir',
    version='0.1.0',
    description='Python package with code described in https://arxiv.org/abs/2008.07231.',
    author='Piotr Masztalski',
    author_email='p.masztalski@samsung.com',
    url='https://github.com/SRPOL-AUI/storir',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.19'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License ',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8'
    ],
)
