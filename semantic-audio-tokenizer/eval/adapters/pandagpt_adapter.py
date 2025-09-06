import os
import sys
from typing import Any, Tuple

import torch

def _die(msg: str):
    raise RuntimeError(
        msg + "\n\n"
        "Setup steps:\n"
        "  git clone https://github.com/dandelin/PandaGPT.git\n"
        "  cd PandaGPT && pip install -r requirements.txt && pip install -e .\n"
        "Then export PANDAGPT_HOME=/abs/path/to/PandaGPT\n"
    )

def load_pandagpt_baseline(device: str = "cuda") -> Tuple[Any, Any, Any]:
    """
    Returns (model, video_preprocess_fn, audio_preprocess_fn).
    This attempts to import PandaGPT from the public repo via PANDAGPT_HOME.
    If your local API differs, adjust the imports and wrappers below.
    """
    home = os.getenv("PANDAGPT_HOME", "").strip()
    if not home or not os.path.isdir(home):
        _die("PANDAGPT_HOME env var not set or path not found.")

    if home not in sys.path:
        sys.path.insert(0, home)

    # --- Import public PandaGPT (heuristics; adjust if API differs)
    try:
        # Common structure in public implementations
        from pandagpt.models.pandagpt import PandaGPT
        from pandagpt.processors.imagebind_processors import (
            build_video_preprocessor, build_audio_preprocessor
        )
    except Exception as e:
        # Fallback heuristics
        try:
            from PandaGPT.models.pandagpt import PandaGPT  # noqa: F401
            from PandaGPT.processors.imagebind_processors import (  # noqa: F401
                build_video_preprocessor, build_audio_preprocessor
            )
        except Exception:
            _die(f"Could not import PandaGPT modules: {e}")

    # --- Build model (adjust kwargs if your fork differs)
    try:
        model = PandaGPT.from_pretrained().to(device).eval()
    except Exception:
        # If no from_pretrained, try a default construct
        try:
            model = PandaGPT().to(device).eval()
        except Exception as e:
            _die(f"Failed to construct PandaGPT model: {e}")

    # --- Preprocessors: return callables that map raw tensors to model-ready features
    try:
        video_pre = build_video_preprocessor(device=device)
        audio_pre = build_audio_preprocessor(device=device)
    except Exception:
        # Provide identity fallbacks if the repo already handles raw tensors internally
        def video_pre(vtensor): return vtensor
        def audio_pre(wav, sr): return wav.unsqueeze(0)  # [1, samples]

    return model, video_pre, audio_pre


class PandaGPTBaselineAdapter:
    """
    Normalizes the baseline PandaGPT call signature:
      generate(prompt=..., video=<video_feats>, audio=<audio_feats>, **gen_args) -> str
    If your model.generate uses different kwargs, change them here.
    """
    def __init__(self, model: Any, video_key: str = "video", audio_key: str = "audio"):
        self.model = model
        self.video_key = video_key
        self.audio_key = audio_key

    @torch.inference_mode()
    def generate(self, prompt, video=None, audio=None, **gen_args) -> str:
        # Most public PandaGPT forks implement .generate or .chat; try both.
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt=prompt, **{self.video_key: video, self.audio_key: audio}, **gen_args)
        if hasattr(self.model, "chat"):
            return self.model.chat(prompt=prompt, **{self.video_key: video, self.audio_key: audio}, **gen_args)
        # Last resort: __call__
        return self.model(prompt=prompt, **{self.video_key: video, self.audio_key: audio}, **gen_args)
