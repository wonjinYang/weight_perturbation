# conftest.py
# pytest configuration file to handle environment setup and path issues

import sys
import os
from pathlib import Path

# Add the src directory to Python path so imports work correctly
project_root = Path(__file__).parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set matplotlib backend to non-interactive for testing
import matplotlib
matplotlib.use('Agg')

# Optional: Add some basic pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Suppress some common warnings during testing
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*FigureCanvasAgg is non-interactive.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Attempting to run cuBLAS.*")

def pytest_sessionstart(session):
    """Print environment info at start of test session."""
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__} available")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("\n❌ PyTorch not available - tests will fail")
        
    try:
        import geomloss
        print(f"✓ geomloss available")
    except ImportError:
        print("❌ geomloss not available")
        
    print(f"✓ Python path includes: {src_path}")

# Optional: Skip tests if torch is not available
def pytest_collection_modifyitems(config, items):
    """Skip tests if required dependencies are not available."""
    try:
        import torch
    except ImportError:
        import pytest
        skip_torch = pytest.mark.skip(reason="PyTorch not available")
        for item in items:
            item.add_marker(skip_torch)