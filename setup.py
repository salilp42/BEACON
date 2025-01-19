from setuptools import setup, find_packages

setup(
    name="beacon-uncertainty",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0"
    ],
    author="Salil Patel",
    author_email="salilp42@github.com",
    description="A lightweight framework for uncertainty quantification in deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/salilp42/BEACON",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
