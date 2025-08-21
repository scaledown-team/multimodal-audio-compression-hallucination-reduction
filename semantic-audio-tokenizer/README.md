Semantic Audio Tokenizer (WIP)

Currently this repo experiments with audio compression and reconstruction using SemantiCodec
.

Setup

Install requirements:

cd overview_and_setup
pip install -r requirements.txt


Install the SemantiCodec inference repo:

cd overview_and_setup/SemantiCodec-inference
pip install -e .


Data

Place input videos in:

data/original/


Processed/reconstructed outputs will be saved to:

data/replaced/

Usage

Run codec on a folder of videos:

python semantic-audio-tokenizer/overview_and_setup/run_codec.py \
    --input data/original \
    --output data/replaced

Notes

Large model checkpoints (>2GB) are not in this repo. They are automatically downloaded from Hugging Face on first run (or can be manually placed in your Hugging Face cache).

