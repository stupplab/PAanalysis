from setuptools import setup

setup(
    name='PAanalysis',
    description='python scripts for analysing MD simulations',
    version='0.0',
    packages=['PAanalysis'],
    package_data={'PAanalysis': ['*']},
    include_package_data=True,
    install_requires=[
        'numpy', 
        'mdtraj', 
        'Cython',
        'freud-analysis',
        'scipy'],
    license='MIT',
)
