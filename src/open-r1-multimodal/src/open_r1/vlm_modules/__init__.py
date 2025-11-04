from .vlm_module import VLMBaseModule
from .qwen_module import Qwen2VLModule
from .internvl_module import InvernVLModule

# Optional GLM support (requires newer transformers with Glm4vForConditionalGeneration)
try:
    from .glm_module import GLMVModule
    __all__ = ["VLMBaseModule", "Qwen2VLModule", "InvernVLModule", "GLMVModule"]
except ImportError:
    GLMVModule = None
    __all__ = ["VLMBaseModule", "Qwen2VLModule", "InvernVLModule"]