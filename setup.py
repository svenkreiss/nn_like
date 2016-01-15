import numpy.distutils.misc_util
# from distutils.core import setup, Extension
from setuptools import setup, Extension

setup(
    ext_modules=[Extension("_nn_like", ["_nn_like.c", "nn_like.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),

    install_requires=['numpy>=1.10'],
)
