"""
Dynamic dependencies for ethnicolr.
This module determines platform-specific dependencies.
"""

import platform

def get_tensorflow_requirement():
    """Get the appropriate TensorFlow requirement based on platform."""
    if platform.machine() == 'aarch64':
        return "tensorflow-aarch64>=2.7.2,<2.16"
    else:
        return "tensorflow>=2.7.2,<2.16"

def get_dependencies():
    """Return the full list of dependencies including platform-specific ones."""
    deps = [
        get_tensorflow_requirement(),
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "h5py>=3.1.0",
        "tqdm>=4.62.0"
    ]
    return deps
