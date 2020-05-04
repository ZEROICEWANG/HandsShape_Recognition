from distutils.core import setup, Extension
from Cython.Build import cythonize
# name表示扩展模块的名称
ext = Extension(name='fill_hole', sources=['fill_hole.pyx'])
setup(ext_modules=cythonize(ext))
