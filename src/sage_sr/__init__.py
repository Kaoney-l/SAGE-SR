"""SAGE-SR: semantic guidance with numerical search evidence."""
from .config import SearchConfig
from .priors import LLMController, StaticController, StructuredPrior

__version__ = "0.1.0"
__all__ = ["SAGESRRegressor", "SearchConfig", "LLMController", "StaticController", "StructuredPrior"]


def __getattr__(name):
    if name == "SAGESRRegressor":
        from .estimator import SAGESRRegressor
        return SAGESRRegressor
    raise AttributeError(name)
