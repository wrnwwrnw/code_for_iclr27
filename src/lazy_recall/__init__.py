"""LazyRecall: decision-risk scoring, lazy eviction and bounded CPU recall."""

from .config import GenerationConfig, RecallConfig

__version__ = "0.3.1"
__all__ = ["GenerationConfig", "RecallConfig", "LazyRecallEngine", "generate", "load_model", "BatchEngine", "generate_batch"]


def __getattr__(name):
    if name in ("BatchEngine", "generate_batch"):
        from . import batching
        return getattr(batching, name)
    if name in ("LazyRecallEngine", "generate", "load_model"):
        from . import generation
        return getattr(generation, name)
    raise AttributeError(name)
