rom setuptools import setup, find_packages
from distutils.core import Extension

DISTNAME = 'coramin'
VERSION = '0.1.0'
EXTENSIONS = []
DESCRIPTION = 'Coramin: Pyomo tools for MINLP'
LONG_DESCRIPTION = open('README.md').read()
AUTHOR = 'Coramin Developers'
MAINTAINER_EMAIL = 'carldlaird@users.noreply.github.com'
LICENSE = 'Revised BSD'
URL = 'https://github.com/Coramin/Coramin'

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['pyomo>=5.6', 'numpy', 'scipy'],
    'scripts': [],
    'include_package_data': True,
}

setup(name=DISTNAME,
      version=VERSION,
      packages=find_packages(),
      ext_modules=EXTENSIONS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)
