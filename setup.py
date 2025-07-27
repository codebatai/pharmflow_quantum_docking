# Copyright 2025 PharmFlow Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PharmFlow: Quantum-Enhanced Molecular Docking Platform
Setup script for package installation
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_py = os.path.join(os.path.dirname(__file__), 'src', 'pharmflow', '__init__.py')
    with open(init_py, 'r') as f:
        content = f.read()
        match = re.search(r'^__version__ = ["\']([^"\']+)["\']', content, re.M)
        if match:
            return match.group(1)
    return "0.1.0"

# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def get_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="pharmflow-quantum-docking",
    version=get_version(),
    author="PharmFlow Development Team",
    author_email="pharmflow-dev@example.com",
    description="Quantum-Enhanced Molecular Docking Platform",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pharmflow-quantum-docking",
    project_urls={
        "Documentation": "https://pharmflow.readthedocs.io/",
        "Bug Reports": "https://github.com/your-username/pharmflow-quantum-docking/issues",
        "Source": "https://github.com/your-username/pharmflow-quantum-docking",
        "Demos": "https://github.com/your-username/pharmflow-quantum-docking/tree/main/demos"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "isort>=5.0",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "jupyter>=1.0",
            "jupyterlab>=3.0"
        ],
        "quantum": [
            "qiskit-aer>=0.11.0",
            "qiskit-ibmq-provider>=0.19.0",
            "qiskit-optimization>=0.4.0",
            "qiskit-machine-learning>=0.5.0"
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "bokeh>=2.4",
            "seaborn>=0.11",
            "matplotlib>=3.5"
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "isort>=5.0",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "jupyter>=1.0",
            "jupyterlab>=3.0",
            "qiskit-aer>=0.11.0",
            "qiskit-ibmq-provider>=0.19.0",
            "qiskit-optimization>=0.4.0",
            "qiskit-machine-learning>=0.5.0",
            "plotly>=5.0",
            "dash>=2.0",
            "bokeh>=2.4",
            "seaborn>=0.11",
            "matplotlib>=3.5"
        ]
    },
    entry_points={
        "console_scripts": [
            "pharmflow=pharmflow.cli.main:main",
            "pharmflow-demo=pharmflow.cli.demo:main",
            "pharmflow-benchmark=pharmflow.cli.benchmark:main",
        ]
    },
    include_package_data=True,
    package_data={
        "pharmflow": [
            "data/*.csv",
            "data/*.json",
            "data/proteins/*.pdb",
            "data/ligands/*.sdf",
            "templates/*.html",
            "config/*.yaml"
        ]
    },
    zip_safe=False,
    keywords=[
        "quantum computing",
        "molecular docking",
        "drug discovery",
        "QAOA",
        "pharmacophore",
        "cheminformatics",
        "quantum chemistry",
        "virtual screening",
        "ADMET",
        "protein-ligand interactions"
    ],
    platforms=["any"],
    license="Apache Software License",
    test_suite="tests",
    tests_require=[
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "pytest-mock>=3.0"
    ]
)
