from setuptools import find_packages, setup

setup(
    name='histogramsampler',
    packages=find_packages(include=['histosampler']),
    version='0.1.0',
    description='Sampling from histogram using normalizing flows',
    author='Jiaxun Lu',
    install_requires=['numpy', 'torch'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)