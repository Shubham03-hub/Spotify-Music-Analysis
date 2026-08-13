from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="spotify-track-performance-intelligence",
    version="1.0.0",
    description=(
        "Track popularity scoring, success-tier classification, and genre "
        "trend forecasting for A&R and marketing decision support."
    ),
    author="Data Science Team",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "spotify-pipeline=main:main",
        ],
    },
)
