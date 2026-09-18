"""Architecture-specific cache routing; scoring and recall remain shared."""

from contextlib import nullcontext

from ..attention import AttentionPatch
from ..cache import LazyRecallCache


def text_config(config):
    getter = getattr(config, "get_text_config", None)
    return getter(decoder=True) if callable(getter) else config


def architecture(config):
    kind = text_config(config).model_type
    if kind in ("llama", "mistral"):
        return kind
    if kind in ("qwen3_5", "qwen3_5_text"):
        return "qwen3_5"
    raise ValueError(f"Unsupported architecture: {kind}; supported: llama, mistral, dense qwen3_5 text")


def check_runtime(config):
    import torch
    import transformers

    architecture(config)
    expected = "5.14.1"
    if transformers.__version__ != expected:
        raise RuntimeError(f"LazyRecall requires transformers=={expected}; found {transformers.__version__}")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError(f"LazyRecall requires torch==2.11.0; found {torch.__version__}")


def batch_rope_type(config):
    parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    return parameters.get("rope_type", parameters.get("type", "default"))


class FullAttentionBackend:
    def __init__(self, model, transport):
        self.view = model
        self.cache = self.model_cache = LazyRecallCache(transport)
        self.patch = AttentionPatch(model)

    def audit_context(self):
        return nullcontext()

    def stats(self):
        return {"architecture": architecture(self.view.config), "linear_state_bytes": 0}

    def close(self):
        self.patch.close()


def create_backend(model, transport=None):
    if architecture(model.config) == "qwen3_5":
        from .qwen35 import Qwen35Backend
        return Qwen35Backend(model, transport)
    return FullAttentionBackend(model, transport)


def management_config(model):
    if architecture(model.config) == "qwen3_5":
        from .qwen35 import FullAttentionView
        return FullAttentionView(model).config
    return model.config
