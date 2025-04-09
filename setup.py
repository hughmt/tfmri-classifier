from setuptools import setup, find_packages

setup(
    name="tfmri_classifier",
    version="0.1",
    packages=find_packages(),  # No `where=` needed anymore
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
)
