import typing as T
import os
import torch
import torchaudio
import whisper_timestamped
import librosa
import numpy as np
from jiwer import wer
from tqdm import tqdm
import json
import whisper

whisper_model = None

def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data

def write_json(my_dict, fname):
    with open(fname, 'w', encoding="utf-8") as outfile:
        json.dump(my_dict, outfile, indent=4, ensure_ascii=False)

libritts_transcription = load_json("/content/libritts_transcription.json")

class TimestampedWhisperWrapper:
    
    
    

    def __init__(self, model: str = "medium", device: str = "cuda") -> None:
        self.resamplers: T.Dict[int, torchaudio.transforms.Resample] = {}
        self.model = whisper_timestamped.load_model(model, device=device)
        self.device = device

    def preprocess_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(self.device)
        waveform_16khz = self.resamplers[sample_rate](waveform)
        waveform_16khz_mono = waveform_16khz.mean(axis=0, keepdims=True)
        return waveform_16khz_mono

    def transcribe(self, waveform: torch.Tensor, sample_rate: int = 44100) -> T.Dict[str, T.Any]:
        waveform = waveform.to(self.device)
        waveform_16khz_mono = self.preprocess_waveform(waveform, sample_rate=sample_rate)
        result = whisper_timestamped.transcribe(
            self.model, waveform_16khz_mono.squeeze(), language="en"
        )
        return result

def get_whisper_timestamped(model: str = "openai/whisper-large-v3", device: str = "cuda") -> TimestampedWhisperWrapper:
    return TimestampedWhisperWrapper(model, device)

def calculate_wer(speechfile, reference_text = None):
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("large-v3").cuda()
    transcription = whisper_model.transcribe(speechfile)
    def preprocess_text(string):
        return "".join([c for c in string if c.isalpha() or c.isdigit() or c == " "]).lower()
    return wer(preprocess_text(reference_text), preprocess_text(transcription["text"]))

def unify_energy(signal1, signal2):
    rms1 = np.sqrt(np.mean(signal1.numpy()**2))
    rms2 = np.sqrt(np.mean(signal2.numpy()**2))
    scale_factor = 0 if rms2 <= 1e-4 else rms1 / rms2
    signal2_scaled = signal2 * scale_factor
    return signal1, signal2_scaled

def load_audio(audio_path, sampling_rate=16000):
    y, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        y = resampler(y)
    return y.squeeze()

def unify_length(audio1, audio2):
    min_length = min(audio1.shape[0], audio2.shape[0])
    return audio1[:min_length], audio2[:min_length]

def compute_mel_distance_torchaudio(audio1_path, audio2_path, sr=16000,
                                    n_fft_list=[32,64,128,256,512,1024,2048],
                                    n_mel_list=[5,10,20,40,80,160,320], hop_length_div=4):
    y1 = load_audio(audio1_path, sampling_rate=sr)
    y2 = load_audio(audio2_path, sampling_rate=sr)
    y1, y2 = unify_length(y1, y2)
    y1, y2 = unify_energy(y1, y2)
    y1 = y1.cuda()
    y2 = y2.cuda()
    mel_distance = 0.0
    for n_fft, n_mels in zip(n_fft_list, n_mel_list):
        hop_length = n_fft // hop_length_div
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                                                            hop_length=hop_length, n_mels=n_mels).cuda()
        mel_spec1 = mel_transform(y1)
        mel_spec2 = mel_transform(y2)
        log_mel_spec1 = torchaudio.transforms.AmplitudeToDB()(mel_spec1)
        log_mel_spec2 = torchaudio.transforms.AmplitudeToDB()(mel_spec2)
        mel_distance += torch.mean(torch.abs(log_mel_spec1 - log_mel_spec2))
    mel_distance /= len(n_fft_list)
    return mel_distance.item()

def compute_stft_distance_torchaudio(audio1_path, audio2_path, sr=16000,
                                     n_fft_list=[2048,512]):
    y1 = load_audio(audio1_path, sampling_rate=sr)
    y2 = load_audio(audio2_path, sampling_rate=sr)
    y1, y2 = unify_length(y1, y2)
    y1, y2 = unify_energy(y1, y2)
    y1 = y1.cuda()
    y2 = y2.cuda()
    stft_distance = 0.0
    for n_fft in n_fft_list:
        hop_length = n_fft // 4
        stft1 = torch.stft(y1.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True)
        stft2 = torch.stft(y2.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True)
        log_stft1 = torchaudio.transforms.AmplitudeToDB()(stft1.abs())
        log_stft2 = torchaudio.transforms.AmplitudeToDB()(stft2.abs())
        stft_distance += torch.mean(torch.abs(log_stft1 - log_stft2))
    stft_distance /= len(n_fft_list)
    return stft_distance.item()

def get_audio_duration_and_size(audio_path):
    import soundfile as sf
    info = sf.info(audio_path)
    duration_sec = info.frames / info.samplerate
    size_bytes = os.path.getsize(audio_path)
    return duration_sec, size_bytes

def compute_bitrate(audio_path):
    duration_sec, size_bytes = get_audio_duration_and_size(audio_path)
    if duration_sec == 0:
        return 0
    return (size_bytes * 8) / duration_sec / 1000  # kbps

def compute_compression_ratio(original_path, compressed_path):
    size_orig = os.path.getsize(original_path)
    size_comp = os.path.getsize(compressed_path)
    if size_comp == 0:
        return None
    return size_orig / size_comp

def evaluate_speech(folder_est, folder_gt, file_prefix):
    assert file_prefix in ["libritts", "enh_libritts"]
    filelist_est = sorted([f for f in os.listdir(folder_est) if f.startswith(file_prefix) and f.endswith('.wav')])
    filelist_gt = sorted([f for f in os.listdir(folder_gt) if f.startswith("libritts") and f.endswith('.wav')])
    assert len(filelist_est) == len(filelist_gt)
    mel_distance, stft_distance, wer_scores = [], [], []
    bitrates, compression_ratios = [], []
    for filename in tqdm(filelist_est):
        speech_est = os.path.join(folder_est, filename)
        if file_prefix == "enh_libritts":
            speech_gt = os.path.join(folder_gt, filename.replace("enh_", ""))
            transcription = libritts_transcription[filename.replace("enh_", "")]
        else:
            speech_gt = os.path.join(folder_gt, filename)
            transcription = libritts_transcription[filename]
        if not os.path.exists(speech_est) or not os.path.exists(speech_gt):
            print(f"Skipping missing file pair: {filename}")
            continue
        try:
            mel_d = compute_mel_distance_torchaudio(speech_est, speech_gt)
            stft_d = compute_stft_distance_torchaudio(speech_est, speech_gt)
            wer_score = calculate_wer(speech_est, transcription)
            bitrate = compute_bitrate(speech_est)
            compression_ratio = compute_compression_ratio(speech_gt, speech_est)
            mel_distance.append(mel_d)
            stft_distance.append(stft_d)
            wer_scores.append(wer_score)
            bitrates.append(bitrate)
            if compression_ratio is not None:
                compression_ratios.append(compression_ratio)
            res = {
                "mel_distance": mel_d,
                "stft_distance": stft_d,
                "wer": wer_score,
                "bitrate_kbps": bitrate,
                "compression_ratio": compression_ratio
            }
            write_json(res, speech_est + ".json")
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")
    return {
        "mel_distance": np.mean(mel_distance) if mel_distance else None,
        "stft_distance": np.mean(stft_distance) if stft_distance else None,
        "wer": np.mean(wer_scores) if wer_scores else None,
        "avg_bitrate_kbps": np.mean(bitrates) if bitrates else None,
        "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else None
    }

def evaluate_audio(folder_est, folder_gt, file_prefix):
    assert file_prefix in ["audioset", "musdb"]
    filelist_est = sorted([f for f in os.listdir(folder_est) if f.startswith(file_prefix) and f.endswith(".wav")])
    filelist_gt = sorted([f for f in os.listdir(folder_gt) if f.startswith(file_prefix) and f.endswith(".wav")])
    assert len(filelist_est) == len(filelist_gt)
    mel_distance, stft_distance = [], []
    for filename in tqdm(filelist_est):
        speech_est = os.path.join(folder_est, filename)
        if file_prefix == "musdb":
            speech_gt = os.path.join(folder_gt, filename + "_trimmed.wav")
            if not os.path.exists(speech_gt):
                speech_gt = os.path.join(folder_gt, filename)
        else:
            speech_gt = os.path.join(folder_gt, filename)
        if not os.path.exists(speech_gt):
            print(f"Missing GT for {filename}")
            continue
        try:
            mel_d = compute_mel_distance_torchaudio(speech_est, speech_gt)
            stft_d = compute_stft_distance_torchaudio(speech_est, speech_gt)
            mel_distance.append(mel_d)
            stft_distance.append(stft_d)
            res = {
                "mel_distance": mel_d,
                "stft_distance": stft_d
            }
            write_json(res, speech_est + ".json")
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")
    return {
        "mel_distance": np.mean(mel_distance) if mel_distance else None,
        "stft_distance": np.mean(stft_distance) if stft_distance else None,
    }

# def evaluate_audio(folder_est, folder_gt, file_prefix):
#     assert file_prefix in ["audioset", "musdb"]
#     filelist_est = sorted([f for f in os.listdir(folder_est) if f.startswith(file_prefix) and f.endswith(".wav")])
#     filelist_gt = sorted([f for f in os.listdir(folder_gt) if f.startswith(file_prefix) and f.endswith(".wav")])
#     assert len(filelist_est) == len(filelist_gt)
#     mel_distance, stft_distance = [], []
#     bitrates, compression_ratios = [], []
#     for filename in tqdm(filelist_est):
#         speech_est = os.path.join(folder_est, filename)
#         if file_prefix == "musdb":
#             speech_gt = os.path.join(folder_gt, filename + "_trimmed.wav")
#             if not os.path.exists(speech_gt):
#                 speech_gt = os.path.join(folder_gt, filename)
#         else:
#             speech_gt = os.path.join(folder_gt, filename)

#         if not os.path.exists(speech_gt):
#             print(f"Missing GT for {filename}")
#             continue
#         try:
#             mel_d = compute_mel_distance_torchaudio(speech_est, speech_gt)
#             stft_d = compute_stft_distance_torchaudio(speech_est, speech_gt)
#             bitrate = compute_bitrate(speech_est)
#             compression_ratio = compute_compression_ratio(speech_gt, speech_est)
#             mel_distance.append(mel_d)
#             stft_distance.append(stft_d)
#             bitrates.append(bitrate)
#             if compression_ratio is not None:
#                 compression_ratios.append(compression_ratio)
#             res = {
#                 "mel_distance": mel_d,
#                 "stft_distance": stft_d,
#                 "bitrate_kbps": bitrate,
#                 "compression_ratio": compression_ratio
#             }
#             write_json(res, speech_est + ".json")
#         except Exception as e:
#             print(f"Error evaluating {filename}: {e}")
#     return {
#         "mel_distance": np.mean(mel_distance) if mel_distance else None,
#         "stft_distance": np.mean(stft_distance) if stft_distance else None,
#         "avg_bitrate_kbps": np.mean(bitrates) if bitrates else None,
#         "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else None
#     }


def evaluate_folder(folder_est, folder_gt):
    result_save_path = folder_est + ".json"
    if os.path.exists(result_save_path) and os.path.getsize(result_save_path) > 8:
        print(f"Found existing evaluation result at: {result_save_path}")
        return
    print("Start evaluation", folder_est)
    result = {}
    result["audioset"] = evaluate_audio(folder_est, folder_gt, "audioset")
    result["musdb"] = evaluate_audio(folder_est, folder_gt, "musdb")
    result["speech"] = evaluate_speech(folder_est, folder_gt, "libritts")
    write_json(result, result_save_path)
    print("Done")

if __name__ == "__main__":
    folder_exp = "/content/generated_audio"
    folder_gt = "/content/evaluationset_16k"
    folder_est_list = [folder_exp]
    for folder_est in folder_est_list:
        try:
            evaluate_folder(folder_est.strip(), folder_gt)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("Error processing", folder_est)
