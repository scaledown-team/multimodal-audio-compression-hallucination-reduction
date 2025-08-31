import os
import tempfile
from pathlib import Path
import ffmpeg
import soundfile as sf
from semanticodec import SemantiCodec
import torch

ORIGINAL_DIR = Path("overview_and_setup/data/original")
REPLACED_DIR = Path("overview_and_setup/data/replaced")
TOKEN_RATE = 100
VOCAB_SIZE = 16384

def mp4_to_wav(input_path, output_path, sample_rate=16000, channels=1):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    (ffmpeg
    .input(str(input_path))
    .output(str(output_path), ac=channels, ar=sample_rate)
    .overwrite_output()
    .run(capture_stdout=True, capture_stderr=True))
    print(f"Successfully converted: {input_path.name} -> {output_path.name}")

def process_video_attention(input_path, output_path, token_rate: int = TOKEN_RATE, vocab_size: int = VOCAB_SIZE):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_audio = tmpdir / "extracted.wav"
        processed_audio = tmpdir / "processed.wav"
        
        (ffmpeg
         .input(str(input_path))
         .output(str(extracted_audio), ac=1, ar=16000)
         .overwrite_output()
         .run(quiet=True))
        
        from attention_token_allocation import ModifiedSemantiCodec
        modifiedSemanticodec = ModifiedSemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=vocab_size,
            force_cpu=True
        )
        
        result = modifiedSemanticodec.encode(str(extracted_audio))
        waveform = modifiedSemanticodec.decode(result)
        
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform[0, 0].cpu().numpy() 
        else:
            waveform_np = waveform[0, 0]
        
        sf.write(str(processed_audio), waveform_np, 16000)
        
        video_stream = ffmpeg.input(str(input_path))
        new_audio_stream = ffmpeg.input(str(processed_audio))
        
        (ffmpeg
         .output(
             video_stream.video,
             new_audio_stream.audio,
             str(output_path),
             vcodec="copy",
             acodec="aac",
             shortest=None
         )
         .overwrite_output()
         .run(quiet=True))
    
    print(f"Video processed with attention: {input_path.name} -> {output_path}")

def test_attention():
    ORIGINAL_DIR = Path("overview_and_setup/data/original")
    REPLACED_DIR = Path("overview_and_setup/data/replaced")
    
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {ORIGINAL_DIR.resolve()}")
    
    REPLACED_DIR.mkdir(parents=True, exist_ok=True)
    
    mp4s = sorted(p for p in ORIGINAL_DIR.iterdir() if p.suffix.lower() == ".mp4")
    
    if not mp4s:
        print(f"No .mp4 files found in {ORIGINAL_DIR.resolve()}")
        return
    
    for src in mp4s:
        dst = REPLACED_DIR / f"{src.stem}_attention{src.suffix}"
        process_video_attention(src, dst)
        print(f"Completed: {src.name} -> {dst.name}")

if __name__ == "__main__":
    test_attention()
    mp4_to_wav("overview_and_setup/data/replaced/avhbench_example_attention.mp4", "overview_and_setup/data/replaced/avhbench_example_attention.wav")

