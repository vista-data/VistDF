import os

from setuptools import setup

package = ["vistdf"]
version = "0.0.1"
description = "A dataframe-like interface for processing visual-spatial-temporal data."
author = "Chanwut (Mick) Kittivorawong"
author_email = "chanwutk@gmail.com"

if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()
else:
    long_description = description

install_requires = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as f:
        install_requires = f.read().splitlines()

setup(
    name="vistdf",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    packages=package,
    install_requires=install_requires,
    python_requires=">=3.10",
)
