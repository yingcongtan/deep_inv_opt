# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

from setuptools import setup, find_packages

install_requires = [
    # Intentionally left empty. Prefer conda install of dependencies.
]

tests_require = [
    # Intentionally left empty. Prefer conda install of dependencies.
]

if __name__ == "__main__":
    setup(name="deepinvopt",
          version="0.1.0",
          packages=find_packages(include=["deep_inv_opt"]),
          install_requires=install_requires,
          tests_require=tests_require)
