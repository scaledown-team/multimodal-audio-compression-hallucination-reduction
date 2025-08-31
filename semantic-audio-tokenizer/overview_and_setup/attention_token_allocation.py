import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

from pesq import pesq
from pystoi import stoi

#pip install pesq, pystoi

class ComplexityAnalyzer:
#analyze audio complexity
    def __init__(self, sr=16000):
        self.sr = sr
        
    def analyze_audio(self, audio_path, target_frames):
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)        
        hop_length = max(64, min(512, int(len(audio) / target_frames)))        
        complexity_scores = []
        
        for i in range(target_frames):
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length * 2, len(audio))
            frame = audio[start_sample:end_sample]
            if len(frame) == 0:
                complexity_scores.append(0.0)
                continue
            spectral_complexity = self.spectral_complexity(frame)
            temporal_complexity = self.temporal_complexity(frame)
            entropy_complexity = self.spectral_entropy(frame)
            
            # Weighted combination - weights need empirical validation
            combined_complexity = (
                0.4 * spectral_complexity +    
                0.3 * temporal_complexity +       
                0.3 * entropy_complexity        
            )
            
            complexity_scores.append(combined_complexity)
        
        return np.array(complexity_scores)
    
    def spectral_complexity(self, frame):
        if len(frame) < 64:
            return 0.0
            
        try:
            centroids = librosa.feature.spectral_centroid(
                y=frame, sr=self.sr, hop_length=64
            )[0]
            if len(centroids) > 1:
                complexity = np.std(centroids) / (np.mean(centroids) + 1e-8)
                return min(complexity, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def temporal_complexity(self, frame):
        if len(frame) < 128:
            return 0.0
        try:
            window_size = len(frame) // 4
            zcr_values = []
            for i in range(0, len(frame) - window_size, window_size // 2):
                window = frame[i:i + window_size]
                zcr = librosa.feature.zero_crossing_rate(window)[0][0]
                zcr_values.append(zcr)
            if len(zcr_values) > 1:
                complexity = np.std(zcr_values) / (np.mean(zcr_values) + 1e-8)
                return min(complexity, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def spectral_entropy(self, frame):
        if len(frame) < 64:
            return 0.0
        try:
            freqs, psd = signal.welch(frame, fs=self.sr, nperseg=min(256, len(frame)))            
            psd_norm = psd / (np.sum(psd) + 1e-8)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
            max_entropy = np.log(len(psd_norm))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            return normalized_entropy
        except:
            return 0.0


class QualityMetrics:    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def evaluate_quality(self, reference_audio, degraded_audio):      
        min_len = min(len(reference_audio), len(degraded_audio))
        ref = reference_audio[:min_len]
        deg = degraded_audio[:min_len]
        
        metrics = {}
        
        # PESQ (Perceptual Evaluation of Speech Quality) range 1.0-4.5
        pesq_score = pesq(self.sr, ref, deg, 'wb')
        metrics['pesq'] = {
            'score': pesq_score,
            'interpretation': self.pesq(pesq_score)
        }

        # STOI (Short-Time Objective Intelligibility) range 0.0-1.0
        stoi_score = stoi(ref, deg, self.sr)
        metrics['stoi'] = {
            'score': stoi_score,
            'interpretation': self.stoi(stoi_score)
        }
        
        return metrics
    
    def pesq(self, score):
        if score >= 4.0:
            return "Excellent quality"
        elif score >= 3.0:
            return "Good quality"
        elif score >= 2.0:
            return "Fair quality"
        else:
            return "Poor quality"
    
    def stoi(self, score):
        if score >= 0.9:
            return "High intelligibility"
        elif score >= 0.7:
            return "Good intelligibility"
        elif score >= 0.5:
            return "Fair intelligibility"
        else:
            return "Poor intelligibility"


class AttentionProcessor:
#process tokens using entropy
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.quality_metrics = QualityMetrics()
    
    def process_tokens(self, audio_path, tokens, codec):        
        B, T, C = tokens.shape
        
        complexity_scores = self.complexity_analyzer.analyze_audio(audio_path, T)
        
        #baseline quality
        original_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        baseline_reconstruction = codec.decode(tokens)
        baseline_audio = self._extract_audio_array(baseline_reconstruction)
        
        min_len = min(len(original_audio), len(baseline_audio))
        original_audio = original_audio[:min_len]
        baseline_audio = baseline_audio[:min_len]
        
        baseline_quality = self.quality_metrics.evaluate_quality(original_audio, baseline_audio)
        baseline_pesq = baseline_quality.get('pesq', {}).get('score', 0)
        
        print(f"Baseline PESQ: {baseline_pesq:.3f}")
        
        #entropy-based token selection
        processed_tokens = self._entropy_based_selection(tokens, complexity_scores)        
        processed_reconstruction = codec.decode(processed_tokens)
        processed_audio = self._extract_audio_array(processed_reconstruction)[:min_len]
        processed_quality = self.quality_metrics.evaluate_quality(original_audio, processed_audio)
        processed_pesq = processed_quality.get('pesq', {}).get('score', 0)
        
        improvement = processed_pesq - baseline_pesq
        print(f"Processed PESQ: {processed_pesq:.3f} (change: {improvement:+.3f})")
        
        analysis_data = {
            'complexity_scores': complexity_scores,
            'baseline_quality': baseline_quality,
            'processed_quality': processed_quality,
            'pesq_improvement': improvement
        }
        
        return processed_tokens, analysis_data
    
    def _entropy_based_selection(self, tokens, complexity_scores):
        processed_tokens = tokens.clone()
        
        entropies = []
        window_size = 5
        
        for t in range(len(complexity_scores)):
            start = max(0, t - window_size // 2)
            end = min(len(complexity_scores), t + window_size // 2 + 1)
            
            #calculate entropy of token values in window
            window_tokens = tokens[0, start:end, :].cpu().flatten().numpy()
            unique_tokens, counts = np.unique(window_tokens, return_counts=True)
            probabilities = counts / len(window_tokens)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        
        #low entropy AND low complexity
        low_entropy_threshold = np.percentile(entropies, 30)
        low_complexity_threshold = np.percentile(complexity_scores, 30)
        
        modifications = 0
        for t in range(len(complexity_scores)):
            if (entropies[t] < low_entropy_threshold and 
                complexity_scores[t] < low_complexity_threshold):
                
                for c in range(tokens.shape[2]):
                    current_token = tokens[0, t, c].item()
                    #quantization (2-bit reduction)
                    quantized = ((current_token + 2) // 4) * 4
                    processed_tokens[0, t, c] = min(16383, quantized)
                    if quantized != current_token:
                        modifications += 1

        print(f"Entropy-based modifications: {modifications} tokens")
        return processed_tokens
    
    def _extract_audio_array(self, audio_tensor):
        if isinstance(audio_tensor, torch.Tensor):
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor
        
        return audio_array.flatten()


class ModifiedSemantiCodec(nn.Module):
#Semanticodec modified with dynamic token allocation    
    def __init__(self, token_rate=100, semantic_vocab_size=16384, force_cpu=True):
        super().__init__()
        
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        
        from semanticodec import SemantiCodec as OriginalSemantiCodec
        self.base_codec = OriginalSemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size
        )
        
        self.entropy_processor = AttentionProcessor()
    
    def encode(self, filepath):
        base_tokens = self.base_codec.encode(filepath)
        try:
            processed_tokens, analysis_data = self.entropy_processor.process_tokens(
                filepath, base_tokens, self.base_codec
            )
            return {
                'tokens': processed_tokens,
                'base_tokens': base_tokens,
                'analysis': analysis_data
            }
            
        except Exception as e:
            print(f"processing failed: {e}")
            import traceback
            traceback.print_exc()
            return base_tokens
    
    def decode(self, encoded_data):
        if isinstance(encoded_data, dict):
            tokens = encoded_data.get('tokens', encoded_data.get('base_tokens'))
        else:
            tokens = encoded_data
        
        target_device = next(self.base_codec.encoder.parameters()).device
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        return self.base_codec.decode(tokens)


def test_attention():
    audio_path = "overview_and_setup/data/original/avhbench_example.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
    
    model = ModifiedSemantiCodec(force_cpu=True)
    
    result = model.encode(audio_path)
    
    if isinstance(result, dict) and 'analysis' in result:
        analysis = result['analysis']
        
        print("\nResults:")
        print(f"  Baseline PESQ: {analysis['baseline_quality']['pesq']['score']:.3f}")
        print(f"  Modified PESQ: {analysis['processed_quality']['pesq']['score']:.3f}")
        print(f"  Improvement: {analysis['pesq_improvement']:+.3f}")
        
        print(f"\n  Baseline STOI: {analysis['baseline_quality']['stoi']['score']:.3f}")
        print(f"  Processed STOI: {analysis['processed_quality']['stoi']['score']:.3f}")
        
        if analysis['pesq_improvement'] >= -0.05:  # Within 0.05 PESQ units
            print("\n SUCCESS: Quality preserved within acceptable range")
            return True
        else:
            print(f"\n Quality degradation: {analysis['pesq_improvement']:.3f}")
            return False
    else:
        print("processing failed")
        return False


if __name__ == "__main__":        
    success = test_attention()
