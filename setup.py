from os import path
from setuptools import setup, find_packages
import sys
import versioneer

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='spylind',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Simple definition and fast solution of quantum systems described by a Lindblad master equation.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Morgan Hedges",
    author_email='morgan.hedges@gmail.com',
    url='https://github.com/morgatron/spylind',
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'jax': ['jax', 'jaxlib', 'diffrax', 'equinox'],
        'cyrk': ['CyRK'],
    },
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
