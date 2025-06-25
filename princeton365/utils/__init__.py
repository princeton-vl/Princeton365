"""Utility modules for Princeton365 package."""

# Always available
from . import utils_io

# Optional imports for dev dependencies
def _import_optional_modules():
    modules = {}
    try:
        from . import utils_aruco
        modules['utils_aruco'] = utils_aruco
    except ImportError:
        pass
    
    try:
        from . import utils_depth
        modules['utils_depth'] = utils_depth
    except ImportError:
        pass
    
    try:
        from . import utils_graph
        modules['utils_graph'] = utils_graph
    except ImportError:
        pass
    
    try:
        from . import utils_trajectory
        modules['utils_trajectory'] = utils_trajectory
    except ImportError:
        pass
    
    return modules

# Make optional modules available as attributes
_optional_modules = _import_optional_modules()
for name, module in _optional_modules.items():
    globals()[name] = module

__all__ = ["utils_io"] + list(_optional_modules.keys()) 