# AVHBench Evaluation (PandaGPT Baseline)

This folder contains a **baseline-only** AVHBench evaluation using the public **PandaGPT**.  
SemantiCodec tokens and GAVIE judge are wired as **placeholders** (ready to be enabled later).

## Setup

1. **Install PandaGPT (public)**
   ```bash
   git clone https://github.com/dandelin/PandaGPT.git
   cd PandaGPT
   pip install -r requirements.txt
   pip install -e .

Then export its path:
export PANDAGPT_HOME=/abs/path/to/PandaGPT


2. Install eval dependencies (from your project root)
pip install -r semantic-audio-tokenizer/requirements_avhbench.txt


3. Prepare AVHBench JSONs:
    avhbench_QA.json – list/dict of yes/no items with fields like:
        {"video": "path/to/video.mp4", "text": "Is the dog visible in the video?", "label": "No"}
    avhbench_captions.json – list/dict of caption items with fields like:
        {"video": "path/to/video.mp4", "caption": "A person walks inside while distant traffic noise is heard."}


4. If dataset only has embedded audio in mp4
Run with --use_ffmpeg_audio_extract to auto-extract mono WAV per video using ffmpeg.


EXAMPLE Run:
    export OPENAI_API_KEY=sk-your-key-here

    python semantic-audio-tokenizer/eval/eval_avhbench_pandagpt.py \
    --qa_json /path/to/avhbench_QA.json \
    --cap_json /path/to/avhbench_captions.json \
    --out_dir semantic-audio-tokenizer/results/avhbench \
    --use_ffmpeg_audio_extract \
    --gavie_model gpt-4o-mini \
    --gavie_batch_size 12 --gavie_concurrency 12 \
    --price_in 0.0005 --price_out 0.0015



Output includes:
    Binary QA: Accuracy / Precision / Recall / F1
    Caption: METEOR / CIDEr
    GAVIE-A

Wiring Notes:
    The loader in adapters/pandagpt_adapter.py imports PandaGPT from PANDAGPT_HOME.
    If public fork uses different module paths or .generate(...) kwargs, adjust the adapter only, not the eval script.

    When ready to test SemantiCodec tokens:
        Implement encode(audio_path, tps, stream) in adapters/semcodec_adapter.py
        Add a PandaGPTSemanticAdapter (mirroring the baseline one) and a second model wiring

    Reuse the existing evaluation loops.

    To enable GAVIE (GPT judge) later:
        Replace compute_gavie_scores_placeholder in adapters/gavie_judge.py with OpenAI call


Tips:

    If you hit ImportError for PandaGPT, confirm:
        PANDAGPT_HOME points to the repo root
        pip install -e $PANDAGPT_HOME succeeded

    If you lack ffmpeg, install it (macOS: brew install ffmpeg, Ubuntu: apt-get install ffmpeg).