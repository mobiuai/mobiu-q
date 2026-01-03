"""
Mobiu-Q v3.0.0 - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mobiu-q",
    version="3.0.0",
    author="Mobiu Technologies",
    author_email="support@mobiu.ai",
    description="Soft Algebra Optimizer + O(N) Linear Attention for Long Context LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mobiu-ai/mobiu-q",
    project_urls={
        "Bug Tracker": "https://github.com/mobiu-ai/mobiu-q/issues",
        "Documentation": "https://docs.mobiu.ai",
        "Homepage": "https://mobiu.ai",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "machine-learning",
        "deep-learning",
        "pytorch",
        "optimizer",
        "attention",
        "transformer",
        "linear-attention",
        "long-context",
        "quantum-computing",
        "soft-algebra",
        "llm",
        "nlp",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
        "quantum": [
            "qiskit>=0.36.0",
            "pennylane>=0.28.0",
        ],
    },
)
