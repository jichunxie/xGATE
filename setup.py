from setuptools import setup, find_packages

setup(
    name="xGATE",                  # Name of your package
    version="1.0.0",               # Version number
    packages=find_packages(),      # Automatically find packages in your project
    # Minimum-version floors, not exact pins: this is a library, so it must
    # co-install with a range of environments. Exact reproducibility pins live in
    # the manuscript repo's requirements.txt / environment.yml.
    install_requires=[             # List any package dependencies here
        "numpy>=1.24",
        "pandas>=1.2",
        "statsmodels>=0.14",
        "mygene>=3.2",
        "networkx>=2.5",
        "igraph>=0.11",
        "torch>=2.0",
        "scipy>=1.10",
        "biopython>=1.78",
        "matplotlib>=3.3",
    ],
    description="A pipeline for single cell rna seq pathway analysis",
    url="https://github.com/jichunxie/xGATE",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',       # Specify the Python versions supported
)
