"""Minimal install script for the SPQR benchmark package."""
import os

from setuptools import find_packages, setup


def _read_requirements():
    here = os.path.dirname(os.path.abspath(__file__))
    reqs = []
    with open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def _read_long_description():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="spqr-benchmark",
    version="0.1.0",
    description="SPQR: A Multi-Dimensional Benchmark for Safety Alignment under Benign Model Adaptation",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="Mohammed Talha Alam et al.",
    author_email="mohammed.alam@mbzuai.ac.ae",
    url="https://github.com/talha-alam/spqr",
    license="MIT",
    packages=find_packages(include=["spqr", "spqr.*", "methods", "methods.*"]),
    python_requires=">=3.8",
    install_requires=_read_requirements(),
    entry_points={
        "console_scripts": [
            "spqr-benchmark=scripts.run_benchmark:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
