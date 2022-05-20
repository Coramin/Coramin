from setuptools import setup, find_packages
from distutils.core import Extension


setup(name='coramin',
      version='0.2.1',
      packages=find_packages(),
      ext_modules=[],
      description='Coramin: Pyomo tools for MINLP',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      author='Coramin Developers',
      maintainer_email='mlbynum@sandia.gov',
      license='Revised BSD',
      url='https://github.com/Coramin/Coramin',
      install_requires=['pyomo>6.4.1', 'numpy', 'scipy', 'networkx'],
      include_package_data=True,
      scripts=[],
      python_requires='>=3.7',
      classifiers=["Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent"])
