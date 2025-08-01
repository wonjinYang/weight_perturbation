# This script is used to build and install the weight_perturbation library.
# It defines the package metadata, dependencies, and installation configuration.
# To install the package locally in editable mode, run: pip install -e .
# To create a source distribution: python setup.py sdist
# To upload to PyPI (requires twine): python setup.py sdist bdist_wheel && twine upload dist/*

import os
from setuptools import setup, find_packages

# Read the long description from README.md
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

# Read the version from __init__.py without importing the package
def get_version():
    version_file = os.path.join('src', 'weight_perturbation', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip("'").strip('"')
    raise RuntimeError("Unable to find version string.")

# Define the setup configuration
setup(
    name='weight_perturbation',  # Package name (as it will appear on PyPI)
    version=get_version(),  # Dynamically retrieve version from __init__.py
    author='Your Name',  # Replace with the actual author's name
    author_email='your.email@example.com',  # Replace with the actual email
    description='A modular Python library for Weight Perturbation strategy based on congested transport.',
    long_description=read_long_description(),  # Use README.md as the long description
    long_description_content_type='text/markdown',  # Specify Markdown format for PyPI
    url='https://github.com/yourusername/weight_perturbation',  # Replace with the actual repository URL
    license='MIT',  # License type
    classifiers=[  # Classifiers for PyPI (help with discoverability)
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='weight perturbation optimal transport congested transport gan wgan pytorch',  # Keywords for searchability
    python_requires='>=3.8',  # Minimum Python version required
    packages=find_packages(where='src'),  # Automatically find packages in the 'src' directory
    package_dir={'': 'src'},  # Map the root package to the 'src' directory
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    install_requires=[  # Runtime dependencies (match requirements.txt)
        'torch>=2.0.0',
        'geomloss>=0.2.6',
        'numpy>=1.24.0',
        'matplotlib>=3.8.0',
        'pyyaml>=6.0.1',
        'typing-extensions>=4.5.0',
    ],
    extras_require={  # Optional dependencies (e.g., for development or testing)
        'dev': [
            'pytest>=7.2.0',
            'setuptools>=65.0.0',
            'wheel>=0.38.0',
            'twine>=4.0.0',  # For uploading to PyPI
        ],
    },
    entry_points={  # Optional: Define console scripts if needed (e.g., CLI tools)
        'console_scripts': [
            # Example: 'weight_perturbation_cli = weight_perturbation.cli:main',
        ],
    },
    project_urls={  # Additional links for PyPI
        'Documentation': 'https://weight-perturbation.readthedocs.io/',  # Replace if you have docs
        'Source': 'https://github.com/yourusername/weight_perturbation',
        'Tracker': 'https://github.com/yourusername/weight_perturbation/issues',
    },
    zip_safe=False,  # Avoid issues with zipped eggs
)
