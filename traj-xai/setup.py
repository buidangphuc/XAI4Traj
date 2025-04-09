from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="traj-xai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Phuc Bui Dang",
    author_email="your.email@example.com",
    description="A package for explainable AI on trajectory data",
    keywords="trajectory, xai, explainable ai",
    url="https://github.com/yourusername/traj-xai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)