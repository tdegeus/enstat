
from setuptools import setup
from setuptools import find_packages

import re

filepath = 'enstat/__init__.py'
__version__ = re.findall(r'__version__ = \'(.*)\'', open(filepath).read())[0]

setup(
    name = 'enstat',
    version = __version__,
    license = 'MIT',
    author = 'Tom de Geus',
    author_email = 'tom@geus.me',
    description = 'Ensemble averages',
    long_description = 'Ensemble averages',
    keywords = 'Statistics, Ensemble',
    url = 'https://github.com/tdegeus/enstat',
    packages = find_packages(),
    install_requires = ['numpy'])
