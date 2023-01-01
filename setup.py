from setuptools import setup, find_packages

from coral_deeplab import __version__


def get_long_desc():
    with open("README.md", "r") as file:
        description = file.read()
    return description


setup(
    name="coral_deeplab",
    version=__version__,
    author="xadrianzetx",
    url="https://github.com/xadrianzetx/coral-deeplab",
    description="Coral Edge TPU compilable version of DeepLab v3",
    long_description=get_long_desc(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["coral_deeplab"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=["numpy<=1.24.0", "tensorflow-gpu>=2.4.0"],
)
