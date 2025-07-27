from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pharmflow-quantum-docking",
    version="1.0.0",
    author="PharmFlow/Codebat Technology",
    author_email="contact@pharmflow.ai",
    description="QAOA and Pharmacophore-Optimized Quantum Molecular Docking Solution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pharmflow/quantum-docking",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0.0", "flake8>=6.0.0"],
        "gpu": ["cupy>=12.0.0", "cuquantum>=23.0.0"],
        "docs": ["sphinx>=7.0.0", "sphinx-rtd-theme>=1.3.0"],
    },
    entry_points={
        "console_scripts": [
            "pharmflow-dock=pharmflow.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
