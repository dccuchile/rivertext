#! /usr/bin/env python
"""Rivertext: An Python Library for training and evaluating on Incremental Word.
Embedding"""

import codecs

from setuptools import find_packages, setup

import rivertext

DISTNAME = "rivertext"
DESCRIPTION = (
    "An Python Library for training and evaluating on Incremental Word Embedding."
)
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = "Rivertext Team"
MAINTAINER = "Rivertext Team"
MAINTAINER_EMAIL = "gabrielturrab@ug.uchile.cl"
URL = "https://github.com/dccuchile/rivertext"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/dccuchile/rivertext"
VERSION = rivertext.__version__
INSTALL_REQUIRES = [
    "nltk",
    "numpy",
    "river",
    "scikit_learn",
    "scipy",
    "torch",
    "tqdm",
    "word-embeddings-benchmarks",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

setup(
    name=DISTNAME,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.6",
    include_package_data=True,
)
