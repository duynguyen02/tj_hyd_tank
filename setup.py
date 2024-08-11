from setuptools import find_packages, setup
from codecs import open
from os import path

from tj_hyd_tank import __version__, __author__

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tj_hyd_tank',
    packages=find_packages(include=['tj_hyd_tank']),
    version=__version__,
    description='Python implementation of Tank Hydrological model by Sugawara and Funiyuki (1956)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    install_requires=[
        'attrs==23.2.0',
        'certifi==2024.7.4',
        'cftime==1.6.4',
        'click==8.1.7',
        'click-plugins==1.1.1',
        'cligj==0.7.2',
        'contourpy==1.2.1',
        'cycler==0.12.1',
        'exceptiongroup==1.2.2',
        'fiona==1.9.6',
        'fonttools==4.53.1',
        'importlib_metadata==8.2.0',
        'importlib_resources==6.4.0',
        'iniconfig==2.0.0',
        'kiwisolver==1.4.5',
        'matplotlib==3.9.1',
        'netCDF4==1.7.1.post1',
        'numpy==2.0.1',
        'packaging==24.1',
        'pandas==2.2.2',
        'pathlib==1.0.1',
        'patsy==0.5.6',
        'pillow==10.4.0',
        'pluggy==1.5.0',
        'pyparsing==3.1.2',
        'pyscissor==1.1.7',
        'pytest==8.3.2',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'scipy==1.13.1',
        'seaborn==0.13.2',
        'shapely==2.0.5',
        'six==1.16.0',
        'statsmodels==0.14.2',
        'tabulate==0.9.0',
        'termcolor==2.3.0',
        'tomli==2.0.1',
        'tzdata==2024.1',
        'yaspin==3.0.2',
        'zipp==3.19.2',
    ]
)