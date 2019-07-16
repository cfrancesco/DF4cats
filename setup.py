from setuptools import setup, find_packages

long_description = '''
DF4cats is a small package meant to facilitate Machine Learning pipelines that include
pandas dataframes with categorical variables.
'''

setup(
    name='df4cats',
    version='0.0.5',
    author='Francesco Cardinale',
    author_email='testadicardi@gmail.com',
    description='DataFrames for cats',
    long_description=long_description,
    license='',
    install_requires=['pandas', 'numpy'],
    extras_require={'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1']},
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
)
