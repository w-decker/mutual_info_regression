from setuptools import setup, find_packages

setup(
    name="mutual_info_regression",
    version="0.0.2",
    author="Will Decker",
    author_email="will.decker@gatech.edu",
    description="Estimate mutual information for a continuous target variable using Holmes and Nemenman (2019) estimator",
    url="https://github.com/w-decker/mutual_info_regression",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ]

)