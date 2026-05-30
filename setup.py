"""Editable-install setup for the cognitive-distortions analysis package.

WHY THIS FILE EXISTS
--------------------
Installing the project with ``pip install -e .`` puts ``src`` on the import path
so ``from src...`` imports work from anywhere (notebooks, tests, scripts) without
manual ``sys.path`` hacks or ``PYTHONPATH`` exports.
"""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="cognitive-distortions",
    version="1.0.0",
    description="Tracking cognitive distortions in Reddit communities before/during/after COVID-19.",
    author="Tawseeq Ahmad",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)
