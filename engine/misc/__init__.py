"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import warnings

from .logger import *
from .visualizer import *
from .dist_utils import setup_seed, setup_print

try:
    from .profiler_utils import stats
except Exception as e:
    warnings.warn(
        f"Skip optional profiler import due to environment mismatch: {e}",
        RuntimeWarning,
    )

    def stats(*args, **kwargs):
        raise RuntimeError(
            "FLOPs profiling is unavailable in current environment. "
            "Please install compatible calflops/transformers/torch versions."
        ) from e
