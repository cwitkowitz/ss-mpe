from setuptools import setup, find_packages

setup(
    name='ss-mpe',
    url='https://github.com/cwitkowitz/self-supervised-multi-pitch',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[],
    version='0.0.1',
    license='TODO',
    description='Code for Self-Supervised MPE framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)