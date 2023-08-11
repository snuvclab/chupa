from setuptools import find_packages, setup

setup(
    name='chupa',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)