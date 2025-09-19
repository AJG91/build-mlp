from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='build_mlp',
    version='0.1',
    packages=find_packages(where='build_mlp'),
    package_dir={'': 'build_mlp'},
    py_modules=[splitext(basename(path))[0] for path in glob('build_mlp/*.py')],
    description='Uses PyTorch to train a multi-layer perceptron on the California housing dataset.',
    author='Alberto J. Garcia',
    zip_safe=False
)
