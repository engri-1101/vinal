from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vinal",
    version="0.0.3",
    author="Henry Robbins",
    author_email="hw.robbins@gmail.com",
    description="A Python package for visualizing graph algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henryrobbins/tmp.git",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19',
        'pandas>=1.1',
        'networkx>=2.0',
        'bokeh>=2.2',
        'typing>=3.7',
    ],
    python_requires='>=3.5',
)