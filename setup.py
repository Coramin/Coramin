from setuptools import setup, find_packages
from distutils.core import Extension


setup(name='coramin',
      version='0.1.1.dev0',
      packages=find_packages(),
      ext_modules=[],
      description='Coramin: Pyomo tools for MINLP',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      author='Coramin Developers',
      maintainer_email='mlbynum@sandia.gov',
      license='Revised BSD',
      url='https://github.com/Coramin/Coramin',
      install_requires=['pyomo>=5.6', 'numpy', 'scipy'],
      include_package_data=True,
      scripts=[],
      python_requires='>=3.6',
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent"])
