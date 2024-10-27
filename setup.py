from setuptools import setup, find_packages

setup(
    name="indentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "torch"
    ],
)