from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/cosmostatistics/21cm_pie"

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "scipy", "torch", "FrEIA", "pyyaml", "21cmFAST", "psutil", "getdist"]

setup(
    name="twentyone_cm_pie",
    version="1.0.1",
    author="Benedikt Schosser",
    author_email="schosser@stud.uni-heidelberg.de",
    description="Simulation based inference for 21cm cosmology",
    long_description=long_description,
    long_description_content_type="text/md",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["twentyone_cm_pie=twentyone_cm_pie.__main__:main"]},
)