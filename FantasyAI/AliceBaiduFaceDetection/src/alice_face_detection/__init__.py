"""Alice lightweight face-detection models and reproducible pipeline helpers.

Model classes are loaded lazily so dataset utilities do not require PaddlePaddle.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PPYoloMobileNetV3", "PPYoloTiny"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .PPYoloMobileNetV3 import PPYoloMobileNetV3, PPYoloTiny

        return {
            "PPYoloMobileNetV3": PPYoloMobileNetV3,
            "PPYoloTiny": PPYoloTiny,
        }[name]
    raise AttributeError(name)
