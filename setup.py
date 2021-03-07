from setuptools import setup

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
    classifiers=[],
    python_requires='>=3.7',
    install_requires=[
        'tensorflow>=2.4.0'
    ]
)
