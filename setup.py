from setuptools import setup, find_packages

setup(
    name="xGATE",                  # Name of your package
    version="0.1.0",               # Version number
    packages=find_packages(),      # Automatically find packages in your project
    install_requires=[             # List any package dependencies here
        "numpy==1.24.4",
        "pandas==1.2.4",
        "statsmodels==0.14.1",
        "mygene==3.2.2",
        "networkx==2.5",
        "igraph==0.11.6",
        "torch==2.4.1",
        "scipy==1.10.1",
        "biopython==1.78",
        "matplotlib==3.3.4"
    ],
    author="Orlando Ferrer",            
    author_email="orlando8955@gmail.com",  
    description="A pipeline for single cell rna seq pathway analysis",     
    url="https://github.com/Cafecito95/xGATE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "CC BY-NC-ND 4.0 license", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',       # Specify the Python versions supported
)
