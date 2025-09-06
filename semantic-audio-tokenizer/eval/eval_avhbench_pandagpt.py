#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features
--------
- Baseline PandaGPT (public repo) audio+video → LLM
- Binary QA metrics: Accuracy / Precision / Recall / F1 (Yes is positive)
- Caption metrics: METEOR (NLTK), CIDEr (pycocoevalcap), per-item & corpus means
- GAVIE judge (OpenAI Responses API, structured JSON), batched + cost/throughput
- Optional pairwise judge (A vs B) for future semantic codec path
- tqdm progress bars, logging, dataclasses for detailed rows
- Writes 4 CSVs + 1 JSON bundle
- ffmpeg audio extraction from MP4s when separate WAV not provided

Prereqs
-------
1) Install PandaGPT (public) and set env var:
   git clone https://github.com/dandelin/PandaGPT.git
   cd PandaGPT && pip install -r requirements.txt && pip install -e .
   export PANDAGPT_HOME=/abs/path/to/PandaGPT

2) Project deps:
   pip install torch torchvision torchaudio numpy pillow tqdm pandas decord opencv-python nltk pycocotools
   pip install git+https://github.com/salaniz/pycocoevalcap.git
   # (for METEOR)
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   # (for GAVIE; optional)
   pip install openai

3) AVHBench JSONs:
   --qa_json : list/dict of yes/no items with fields like {"video": "...mp4", "text": "...?", "label": "Yes/No"}
   --cap_json: list/dict with {"video": "...mp4", "caption": "reference"} or {"references":[...]} etc.

Run
---
python semantic-audio-tokenizer/eval/eval_avhbench_pandagpt.py \
  --qa_json /path/to/avhbench_QA.json \
  --cap_json /path/to/avhbench_captions.json \
  --out_dir semantic-audio-tokenizer/results/avhbench \
  --use_ffmpeg_audio_extract \
  --gavie_model gpt-4o-mini \
  --gavie_batch_size 12 --gavie_concurrency 12 \
  --price_in 0.0005 --price_out 0.0015

Set OPENAI_API_KEY in your environment for GAVIE scoring.
"""

import os, sys, json, time, argparse, subprocess
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
from PIL import Image
import torch, torchaudio
from tqdm import tqdm
import pandas as pd

# Optional fast video decode
try:
    import decord
    decord.bridge.set_bridge('torch')
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    from torchvision.io import read_video
except Exception:
    read_video = None

# Caption metrics
try:
    from nltk.translate.meteor_score import meteor_score
except Exception:
    meteor_score = None

# ---- adapters path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTERS_DIR = os.path.join(THIS_DIR, "adapters")
if ADAPTERS_DIR not in sys.path:
    sys.path.insert(0, ADAPTERS_DIR)

# Import GAVIE (kept in separate file)
from gavie_judge import (
    compute_gavie_scores_openai_batched,
    compute_gavie_pairwise_openai_batched
)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("avhbench.eval")

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class QARecord:
    qid: str
    task: str
    question: str
    gt_yes: bool
    pred_yes: bool

@dataclass
class CapRecord:
    qid: str
    ref: str
    hyp: str
    meteor: Optional[float] = None
    cider: Optional[float] = None

# -------------------------
# Registry
# -------------------------
MODEL_REGISTRY: Dict[str, Any] = {
    "pandagpt_baseline": None,
    "video_preprocess": None,
    "audio_preprocess": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "frame_size": 224,
    "num_frames": 8,
    "audio_sr": 16000,
    "use_ffmpeg_audio_extract": False,
    "ffmpeg_bin": "ffmpeg",
}
def set_model_registry(**kwargs): MODEL_REGISTRY.update(kwargs)

# -------------------------
# PandaGPT loader/adapter (inline)
# -------------------------
def _die(msg: str):
    raise RuntimeError(
        msg + "\n\n"
        "Setup:\n"
        "  git clone https://github.com/dandelin/PandaGPT.git\n"
        "  cd PandaGPT && pip install -r requirements.txt && pip install -e .\n"
        "  export PANDAGPT_HOME=/abs/path/to/PandaGPT\n"
    )

def load_pandagpt_baseline(device: str = "cuda"):
    home = os.getenv("PANDAGPT_HOME", "").strip()
    if not home or not os.path.isdir(home):
        _die("PANDAGPT_HOME env var not set or path not found.")
    if home not in sys.path:
        sys.path.insert(0, home)

    PandaGPT = None
    build_video_preprocessor = build_audio_preprocessor = None
    last_err = None
    for modtry in [
        ("pandagpt.models.pandagpt", "pandagpt.processors.imagebind_processors"),
        ("PandaGPT.models.pandagpt", "PandaGPT.processors.imagebind_processors"),
        ("pandagpt.model", "pandagpt.processors"),
    ]:
        try:
            mod_model = __import__(modtry[0], fromlist=["PandaGPT"])
            mod_proc  = __import__(modtry[1], fromlist=["build_video_preprocessor","build_audio_preprocessor"])
            PandaGPT = getattr(mod_model, "PandaGPT")
            build_video_preprocessor = getattr(mod_proc, "build_video_preprocessor", None)
            build_audio_preprocessor = getattr(mod_proc, "build_audio_preprocessor", None)
            break
        except Exception as e:
            last_err = e
            continue
    if PandaGPT is None:
        _die(f"Could not import PandaGPT: {last_err}")

    try:
        model = PandaGPT.from_pretrained().to(device).eval()
    except Exception:
        try:
            model = PandaGPT().to(device).eval()
        except Exception as e:
            _die(f"Failed to construct PandaGPT: {e}")

    if callable(build_video_preprocessor) and callable(build_audio_preprocessor):
        try:
            video_pre = build_video_preprocessor(device=device)
            audio_pre = build_audio_preprocessor(device=device)
        except Exception:
            def video_pre(v): return v
            def audio_pre(wav, sr): return wav.unsqueeze(0)
    else:
        def video_pre(v): return v
        def audio_pre(wav, sr): return wav.unsqueeze(0)

    class PandaGPTBaselineAdapter:
        def __init__(self, model, video_key="video", audio_key="audio"):
            self.model = model; self.vk = video_key; self.ak = audio_key
        @torch.inference_mode()
        def generate(self, prompt, video=None, audio=None, **gen_args) -> str:
            if hasattr(self.model, "generate"):
                return self.model.generate(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)
            if hasattr(self.model, "chat"):
                return self.model.chat(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)
            return self.model(prompt=prompt, **{self.vk: video, self.ak: audio}, **gen_args)

    return PandaGPTBaselineAdapter(model), video_pre, audio_pre

# -------------------------
# Video / Audio I/O
# -------------------------
def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == h: return img.resize((size, size), Image.BICUBIC)
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - size)//2, (new_h - size)//2
    return img.crop((left, top, left + size, top + size))

def _to_tensor(frames: List[Image.Image], dtype: torch.dtype) -> torch.Tensor:
    arr = np.stack([np.asarray(f).astype(np.float32) / 255.0 for f in frames], axis=0)
    arr = np.transpose(arr, (0,3,1,2))
    t = torch.from_numpy(arr).to(dtype=dtype)
    return t.unsqueeze(0)  # [1,T,C,H,W]

def load_video_frames(video_path: str, num_frames: int, size: int) -> List[Image.Image]:
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    frames: List[Image.Image] = []
    if HAS_DECORD:
        vr = decord.VideoReader(video_path)
        total = len(vr); assert total > 0, f"No frames in {video_path}"
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        batch = vr.get_batch(idx)
        for i in range(batch.shape[0]):
            frames.append(_center_crop(Image.fromarray(batch[i].cpu().numpy()).convert("RGB"), size))
        return frames
    if read_video is None:
        raise RuntimeError("Neither decord nor torchvision.io.read_video available.")
    vid, _, _ = read_video(video_path, pts_unit='sec')
    total = vid.shape[0]; assert total > 0, f"No frames in {video_path}"
    idx = np.linspace(0, total - 1, num_frames).astype(int)
    for i in idx:
        frames.append(_center_crop(Image.fromarray(vid[i].numpy()).convert("RGB"), size))
    return frames

@lru_cache(maxsize=128)
def _cached_video_tensor(video_path: str, num_frames: int, size: int, dtype_name: str):
    frames = load_video_frames(video_path, num_frames=num_frames, size=size)
    return _to_tensor(frames, dtype=getattr(torch, dtype_name))

def _ffmpeg_available(bin_name: str) -> bool:
    try:
        subprocess.run([bin_name, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def _ffmpeg_extract_audio(video_path: str, out_wav: str, sr: int, ffmpeg_bin: str = "ffmpeg") -> bool:
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", out_wav]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def load_audio_waveform(audio_path: str, target_sr: int):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr); sr = target_sr
    return wav.squeeze(0), sr

@lru_cache(maxsize=256)
def _cached_audio_waveform(audio_path: str, target_sr: int):
    return load_audio_waveform(audio_path, target_sr)

# -------------------------
# Inference
# -------------------------
@torch.inference_mode()
def multimodal_inference_baseline(audio_path: Optional[str], video_path: Optional[str], prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None) -> str:
    device = MODEL_REGISTRY["device"]; dtype = MODEL_REGISTRY["dtype"]
    size = int(MODEL_REGISTRY.get("frame_size", 224))
    num_frames = int(MODEL_REGISTRY.get("num_frames", 8))
    target_sr = int(MODEL_REGISTRY.get("audio_sr", 16000))
    use_ffmpeg = bool(MODEL_REGISTRY.get("use_ffmpeg_audio_extract", False))
    ffmpeg_bin = MODEL_REGISTRY.get("ffmpeg_bin", "ffmpeg")

    gen_args = dict(max_new_tokens=64, temperature=0.0, top_p=1.0, do_sample=False)
    if generation_kwargs: gen_args.update(generation_kwargs)

    vid_feats = None
    if video_path:
        vtensor = _cached_video_tensor(video_path, num_frames, size, dtype_name=str(dtype).split('.')[-1]).to(device)
        vid_feats = MODEL_REGISTRY["video_preprocess"](vtensor) if callable(MODEL_REGISTRY["video_preprocess"]) else vtensor

    audio_feats = None
    if audio_path is None and video_path and use_ffmpeg and _ffmpeg_available(ffmpeg_bin):
        tmp_wav = os.path.join(".avhbench_tmp_audio", os.path.basename(video_path) + ".wav")
        if _ffmpeg_extract_audio(video_path, tmp_wav, target_sr, ffmpeg_bin):
            audio_path = tmp_wav
    if audio_path and os.path.exists(audio_path):
        wav, sr = _cached_audio_waveform(audio_path, target_sr)
        wav = wav.to(device=device, dtype=torch.float32)
        audio_feats = MODEL_REGISTRY["audio_preprocess"](wav, sr)
        if isinstance(audio_feats, torch.Tensor):
            audio_feats = audio_feats.to(device)

    out_text = MODEL_REGISTRY["pandagpt_baseline"].generate(prompt=prompt, video=vid_feats, audio=audio_feats, **gen_args)
    return str(out_text).strip()

# -------------------------
# Metrics
# -------------------------
def compute_classification_metrics(gt_labels: List[bool], pred_labels: List[bool]):
    assert len(gt_labels) == len(pred_labels)
    total = len(gt_labels)
    correct = sum(int(g == p) for g, p in zip(gt_labels, pred_labels))
    acc = correct / total if total else 0.0
    TP = sum(1 for g, p in zip(gt_labels, pred_labels) if p and g)
    FP = sum(1 for g, p in zip(gt_labels, pred_labels) if p and not g)
    FN = sum(1 for g, p in zip(gt_labels, pred_labels) if (not p) and g)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return acc, prec, rec, f1

def compute_caption_overlap_metrics(ref_lists: List[List[str]], hypotheses: List[str]):
    """
    Returns (meteor_mean, cider_mean, meteor_per_item, cider_per_item).
    """
    n = len(hypotheses)
    meteor_item: List[Optional[float]] = [None] * n
    if meteor_score is not None and n > 0:
        for i, (refs, hyp) in enumerate(zip(ref_lists, hypotheses)):
            try: meteor_item[i] = float(meteor_score(refs, hyp))
            except Exception: meteor_item[i] = None
        meteor_mean = (sum(x for x in meteor_item if x is not None) /
                       max(1, sum(x is not None for x in meteor_item))) if meteor_item else None
    else:
        meteor_mean = None

    cider_item: List[Optional[float]] = [None] * n
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        refs = {i: ref_lists[i] for i in range(n)}
        hyps = {i: [hypotheses[i]] for i in range(n)}
        cider_mean, scores = scorer.compute_score(refs, hyps)
        for i, s in enumerate(scores):
            cider_item[i] = float(s)
    except Exception:
        cider_mean = None

    return meteor_mean, cider_mean, meteor_item, cider_item

# -------------------------
# Dataset utils & evaluation loops
# -------------------------
def normalize_yesno(text: str) -> Optional[bool]:
    if text is None: return None
    t = text.strip().lower()
    if t.startswith("yes"): return True
    if t.startswith("no"):  return False
    return None

def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_qa_by_type(qa_data) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = {}
    def pick_type(item: dict) -> str:
        t = item.get("task") or item.get("task_type") or ""
        if t: return t
        q = (item.get("text") or item.get("question") or "").lower()
        if "visible in the video" in q: return "Audio-driven Video Hallucination"
        if "making sound in the audio" in q: return "Video-driven Audio Hallucination"
        if "matching" in q: return "Audio-Visual Matching"
        return "Unknown"
    if isinstance(qa_data, dict):
        if any(k in qa_data for k in ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","Audio-Visual Matching"]):
            for k, v in qa_data.items(): buckets[k] = v if isinstance(v, list) else [v]
        elif "tasks" in qa_data and isinstance(qa_data["tasks"], list):
            for it in qa_data["tasks"]: buckets.setdefault(pick_type(it), []).append(it)
        else:
            if isinstance(qa_data.get("data"), list):
                for it in qa_data["data"]: buckets.setdefault(pick_type(it), []).append(it)
            else:
                lst = qa_data if isinstance(qa_data, list) else [qa_data]
                for it in lst: buckets.setdefault(pick_type(it), []).append(it)
    elif isinstance(qa_data, list):
        for it in qa_data: buckets.setdefault(pick_type(it), []).append(it)
    else:
        buckets["Unknown"] = []
    return buckets

def evaluate_qa(tasks: List[dict]):
    gt: List[bool] = []; preds: List[bool] = []; rows: List[QARecord] = []
    for idx, item in enumerate(tqdm(tasks, desc="QA inference (baseline)")):
        video_path = item.get("video") or item.get("video_path")
        audio_path = item.get("audio") or item.get("audio_path")
        q = item.get("text") or item.get("question") or ""
        qid = str(item.get("id", f"qa_{idx}"))
        gt_ans = item.get("label") or item.get("answer")
        if isinstance(gt_ans, bool): gt_label = gt_ans
        elif isinstance(gt_ans, str): gt_label = gt_ans.strip().lower() == "yes"
        else: gt_label = bool(gt_ans)
        out = multimodal_inference_baseline(audio_path, video_path, q)
        yn = normalize_yesno(out) or (not gt_label)  # unclear → count as incorrect by flipping
        gt.append(gt_label); preds.append(yn)
        rows.append(QARecord(qid=qid, task=item.get("task") or item.get("task_type") or "", question=q,
                             gt_yes=gt_label, pred_yes=yn))
    return gt, preds, rows

def evaluate_captions(caps: List[dict]):
    ref_lists: List[List[str]] = []; hyps: List[str] = []; qids: List[str] = []; rows: List[CapRecord] = []
    for i, item in enumerate(tqdm(caps, desc="Caption generation (baseline)")):
        video_path = item.get("video") or item.get("video_path")
        audio_path = item.get("audio") or item.get("audio_path")
        refs: List[str] = []
        if "caption" in item and isinstance(item["caption"], str):
            refs = [item["caption"]]
        else:
            for k in ("ground_truth", "reference", "references"):
                v = item.get(k)
                if v:
                    refs = v if isinstance(v, list) else [str(v)]
                    break
        if not refs: continue
        prompt = "Describe what you see and hear."
        out = multimodal_inference_baseline(audio_path, video_path, prompt)
        ref_lists.append([str(r) for r in refs])
        hyps.append(str(out) if out is not None else "")
        qid = str(item.get("id", f"cap_{i}")); qids.append(qid)
        rows.append(CapRecord(qid=qid, ref=refs[0], hyp=hyps[-1]))
    return ref_lists, hyps, qids, rows

# -------------------------
# Persistence
# -------------------------
def save_eval_outputs(out_dir: Path, qa_metrics: Dict[str, Any], qa_rows: List[QARecord],
                      cap_metrics: Dict[str, Any], cap_rows: List[CapRecord]):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([qa_metrics]).to_csv(out_dir / f"binary_metrics_{ts}.csv", index=False)
    if qa_rows: pd.DataFrame([asdict(r) for r in qa_rows]).to_csv(out_dir / f"binary_detailed_{ts}.csv", index=False)
    pd.DataFrame([cap_metrics]).to_csv(out_dir / f"caption_metrics_{ts}.csv", index=False)
    if cap_rows: pd.DataFrame([asdict(r) for r in cap_rows]).to_csv(out_dir / f"caption_detailed_{ts}.csv", index=False)
    bundle = {
        "timestamp": datetime.now().isoformat(),
        "binary_evaluation": qa_metrics,
        "caption_evaluation": cap_metrics,
        "binary": [asdict(r) for r in qa_rows],
        "captions": [asdict(r) for r in cap_rows],
    }
    with open(out_dir / f"eval_results_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {out_dir} (timestamp={ts})")

# -------------------------
# CLI / main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="AVHBench eval — PandaGPT baseline (imports GAVIE from adapters)")
    ap.add_argument("--qa_json", type=str, required=True)
    ap.add_argument("--cap_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="semantic-audio-tokenizer/results/avhbench")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--use_ffmpeg_audio_extract", action="store_true")
    ap.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    # GAVIE
    ap.add_argument("--gavie_model", type=str, default="", help="OpenAI model for GAVIE-A (e.g., gpt-4o-mini). If empty, GAVIE is skipped.")
    ap.add_argument("--gavie_pairwise", action="store_true", help="Also run pairwise judge (requires two hypothesis sets).")
    ap.add_argument("--gavie_batch_size", type=int, default=8)
    ap.add_argument("--gavie_concurrency", type=int, default=8)
    ap.add_argument("--price_in", type=float, default=0.0)
    ap.add_argument("--price_out", type=float, default=0.0)
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Wire PandaGPT
    device = MODEL_REGISTRY["device"]
    baseline_model, video_pre, audio_pre = load_pandagpt_baseline(device=device)
    set_model_registry(
        pandagpt_baseline=baseline_model,
        video_preprocess=video_pre,
        audio_preprocess=audio_pre,
        frame_size=args.frame_size,
        num_frames=args.num_frames,
        audio_sr=args.audio_sr,
        use_ffmpeg_audio_extract=args.use_ffmpeg_audio_extract,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    logger.info("PandaGPT baseline wired. Beginning evaluation...")

    qa_raw  = load_json_any(args.qa_json)
    caps_raw = load_json_any(args.cap_json)
    qa_buckets = group_qa_by_type(qa_raw)

    # QA
    task_order = ["Audio-driven Video Hallucination", "Video-driven Audio Hallucination", "Audio-Visual Matching"]
    qa_rows_all: List[QARecord] = []
    qa_metrics_all: Dict[str, Any] = {}
    logger.info("Running QA (binary) tasks...")
    for key in task_order + [k for k in qa_buckets.keys() if k not in task_order]:
        tasks = qa_buckets.get(key, [])
        if not tasks: continue
        logger.info(f"[Task] {key} (n={len(tasks)})")
        gt, pr, rows = evaluate_qa(tasks)
        acc, prec, rec, f1 = compute_classification_metrics(gt, pr)
        qa_metrics_all[f"{key}/acc"] = acc; qa_metrics_all[f"{key}/prec"] = prec
        qa_metrics_all[f"{key}/rec"] = rec; qa_metrics_all[f"{key}/f1"] = f1
        qa_rows_all.extend(rows)
        logger.info(f"  Acc {acc*100:.1f} | P {prec*100:.1f} | R {rec*100:.1f} | F1 {f1*100:.1f}")

    # Captions
    if isinstance(caps_raw, dict) and "captions" in caps_raw: caps_list = caps_raw["captions"]
    elif isinstance(caps_raw, dict) and not isinstance(next(iter(caps_raw.values())), dict): caps_list = list(caps_raw.values())
    elif isinstance(caps_raw, dict): caps_list = [v for v in caps_raw.values()]
    elif isinstance(caps_raw, list): caps_list = caps_raw
    else: caps_list = [caps_raw]

    logger.info("Generating captions...")
    ref_lists, hyps, qids, cap_rows = evaluate_captions(caps_list)
    logger.info("Scoring captions (METEOR, CIDEr)...")
    meteor_mean, cider_mean, meteor_item, cider_item = compute_caption_overlap_metrics(ref_lists, hyps)
    qid_to_ix = {qids[i]: i for i in range(len(qids))}
    for r in cap_rows:
        ix = qid_to_ix[r.qid]; r.meteor = meteor_item[ix]; r.cider = cider_item[ix]
    cap_metrics = {"meteor_mean": meteor_mean, "cider_mean": cider_mean, "num_captions": len(cap_rows)}
    logger.info(f"Caption metrics: METEOR {meteor_mean if meteor_mean is not None else 'N/A'} | CIDEr {cider_mean if cider_mean is not None else 'N/A'}")

    # QA macro
    accs  = [v for k,v in qa_metrics_all.items() if k.endswith("/acc")]
    precs = [v for k,v in qa_metrics_all.items() if k.endswith("/prec")]
    recs  = [v for k,v in qa_metrics_all.items() if k.endswith("/rec")]
    f1s   = [v for k,v in qa_metrics_all.items() if k.endswith("/f1")]
    qa_overall = {
        "qa_macro_acc":  sum(accs)/len(accs) if accs else None,
        "qa_macro_prec": sum(precs)/len(precs) if precs else None,
        "qa_macro_rec":  sum(recs)/len(recs) if recs else None,
        "qa_macro_f1":   sum(f1s)/len(f1s) if f1s else None,
        "qa_items": len(qa_rows_all), **qa_metrics_all
    }

    # GAVIE (OpenAI) — optional
    gavie_model = args.gavie_model.strip() or None
    if gavie_model and cap_rows:
        refs_for_judge = [r.ref for r in cap_rows]
        hyps_for_judge = [r.hyp for r in cap_rows]
        logger.info("[GAVIE-A] judging captions via OpenAI...")
        gav_avg, gav_list, gav_stats = compute_gavie_scores_openai_batched(
            refs_for_judge, hyps_for_judge,
            model=gavie_model,
            batch_size=args.gavie_batch_size,
            concurrency=args.gavie_concurrency,
            price_in_per_1k=args.price_in,
            price_out_per_1k=args.price_out,
        )
        cap_metrics.update({
            "gavie_a_mean": gav_avg,
            "gavie_a_items": len(gav_list) if gav_list else 0,
            "gavie_tokens_in": gav_stats.get("total_input_tokens", 0.0),
            "gavie_tokens_out": gav_stats.get("total_output_tokens", 0.0),
            "gavie_cost_usd": gav_stats.get("est_cost_usd", 0.0),
            "gavie_items_per_sec": gav_stats.get("items_per_sec", 0.0),
        })

    # Persist
    save_eval_outputs(Path(args.out_dir), qa_overall, qa_rows_all, cap_metrics, cap_rows)
    logger.info("Done.")

if __name__ == "__main__":
    main()