from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vinal",
    version="0.0.5",
    author="Henry Robbins",
    author_email="hw.robbins@gmail.com",
    description="A Python package for visualizing graph algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henryrobbins/tmp.git",
    packages=find_packages(),
    include_package_data=True,
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19',
        'pandas>=1.2',
        'networkx>=3',
        'bokeh>=3',
        'typing>=3.7',
    ],
    python_requires='>=3.8',
)