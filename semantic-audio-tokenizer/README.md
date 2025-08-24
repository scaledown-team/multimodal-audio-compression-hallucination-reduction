# Semantic Audio Tokenizer

Currently this repo experiments with audio compression and reconstruction using **SemantiCodec**.

---

## Setup

### Install requirements:

```bash
cd overview_and_setup
pip install -r requirements.txt

```

### Install the SemantiCodec inference repo:

```bash
cd overview_and_setup/SemantiCodec-inference
pip install -e .
```
### Data
```
Place input videos in:

data/original/

Processed/reconstructed outputs will be saved to:

data/replaced/
```

### Usage
```bash
python semantic-audio-tokenizer/overview_and_setup/run_codec.py \
    --input data/original \
    --output data/replaced
```
### Notes
```
Large model checkpoints (>2GB) are not in this repo.
They are automatically downloaded from Hugging Face on first run (or can be manually placed in your Hugging Face cache).
```
### Evaluation Dataset
```bash
!wget https://zenodo.org/records/11047204/files/semanticodec_evaluationset_16k.tar?download=1 -O semanticodec_evaluationset_16k.tar
!tar -xvf semanticodec_evaluationset_16k.tar
```
### Generate Audio using pretrained checkpoints
```bash
import soundfile as sf
from semanticodec import SemantiCodec

# Initialize the codec with desired settings (example: 1.35 kbps)
semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=16384)
import os

input_folder = "/content/evaluationset_16k"
output_folder = "/content/generated_audio"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        tokens = semanticodec.encode(input_path)
        waveform = semanticodec.decode(tokens)
        output_path = os.path.join(output_folder, filename)
        sf.write(output_path, waveform[0, 0], 16000)
        print(f"Processed {filename}")
```
