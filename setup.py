from setuptools import setup, find_packages

setup(
    name="pulse-library",
    version="2.0.0",
    description="PULSE Tool Factory and Economics System",
    author="jetgause",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "dataclasses-json>=0.6.0",
        "typing-extensions>=4.8.0",
        "pytest>=7.4.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
    ],
    python_requires=">=3.8",
)
