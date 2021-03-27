from setuptools import setup, find_packages

from coral_deeplab import __version__


def get_long_desc():
    with open('README.md', 'r') as file:
        description = file.read()
    return description


setup(
    name='coral_deeplab',
    version=__version__,
    author='xadrianzetx',
    url='https://github.com/xadrianzetx/coral-deeplab',
    description='Coral Edge TPU compilable version of DeepLab v3 Plus',
    long_description=get_long_desc(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['coral_deeplab']),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.5',
        'tensorflow-gpu>=2.4.0'
    ]
)
