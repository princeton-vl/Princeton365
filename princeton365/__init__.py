
try:
    from ._version import __version__
except ImportError:
    # Fallback version for development installations
    __version__ = "0.0.1+dev"

# Import main classes and functions that don't require heavy dependencies
try:
    from .board_generator import generate_charuco_boards, create_board
except ImportError:
    pass

try:
    from .generate_gt import Princeton365
except ImportError:
    pass

try:
    from .evaluation import (
        compute_induced_optical_flow,
        calculate_flow_auc,
    )
except ImportError:
    pass

# Make key utilities accessible at package level (only basic ones for core package)
try:
    from .utils import utils_io
except ImportError:
    pass

__all__ = [
    "generate_charuco_boards",
    "create_board", 
    "Princeton365",
    "compute_induced_optical_flow",
    "calculate_flow_auc",
    "utils_io", 
]
