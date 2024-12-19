from setuptools import setup, find_packages

setup(
    name="epam",
    version="0.1.0",
    url="https://github.com/matsengrp/epam.git",
    author="Matsen Group",
    author_email="ematsen@gmail.com",
    description="Evaluating predictions of affinity maturation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib >= 3.4.3",
        "pandas >= 1.3.3",
        "biopython >= 1.79",
        "seaborn",
        "statsmodels",
        "fire",
        "tables",
    ],
    python_requires=">=3.9,<3.13", 
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": ["epam=epam.cli:main"],
    },
)
