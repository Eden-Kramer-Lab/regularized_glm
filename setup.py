#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ["numpy", "statsmodels", "scipy"]
TESTS_REQUIRE = ["pytest >= 2.7.1"]

setup(
    name="regularized_glm",
    version="1.0.2",
    license="MIT",
    description=("L2-penalized generalized linear models"),
    author="Eric Denovellis",
    author_email="edeno@bu.edu",
    url="https://github.com/Eden-Kramer-Lab/regularized_glm",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
