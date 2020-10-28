from distutils.core import setup, Extension
import numpy

module = Extension("cgal_functions_Module",
                   sources = ["cgal_functions.cpp"],
                   extra_compile_args=['-v','-std=c++0x'],
                   extra_link_args=['-L /usr/include/','-lCGAL','-lgmp','-std=c++0x'],
		   include_dirs=[numpy.get_include()])

setup(name="CGAL_functions",
      version = "1.0",
      description = "This is a package for cgal_functions_Module",
      ext_modules = [module] )
