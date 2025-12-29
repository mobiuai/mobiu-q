from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mobiu-q",
    version="2.7.7",
    author="Mobiu Technologies",
    author_email="contact@mobiu.ai",
    description="Soft Algebra Optimizer for Quantum, RL, LLM & Complex Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://mobiu.ai",
    project_urls={
        "Documentation": "https://docs.mobiu.ai",
        "Source": "https://github.com/mobiu-ai/mobiu-q",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "quantum": ["qiskit>=0.45.0", "qiskit-aer>=0.12.0"],
        "dev": ["pytest", "black", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "mobiu-q=mobiu_q.core:main",
        ],
    },
    keywords=[
        "optimization",
        "quantum computing",
        "VQE",
        "QAOA",
        "reinforcement learning",
        "LLM",
        "fine-tuning",
        "soft algebra",
        "noise robust",
    ],
)