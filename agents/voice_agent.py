# Updated voice_agent.py with real audio processing
from services.openai_services import OpenAIService
from services.speech_services import SpeechService
from .base_agent import BaseAgent
import librosa
import numpy as np
import scipy.signal
from scipy.stats import skew, kurtosis
import parselmouth
from parselmouth.praat import call
import tempfile
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class VoiceAgent(BaseAgent):
    """Agent for processing voice/audio medical data with real audio analysis"""
    
    def __init__(self, agent_id: str = "voice_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.openai_service = OpenAIService()
        self.speech_service = SpeechService()
        
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
    
    async def _analyze_voice_characteristics(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyze voice characteristics for medical indicators using audio processing libraries"""
        try:

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load audio with librosa
                y, sr = librosa.load(temp_file_path, sr=self.sample_rate)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Load with parselmouth for advanced analysis
                sound = parselmouth.Sound(temp_file_path)
                
                # Analyze various voice characteristics
                analysis_results = {
                    'duration': duration,
                    'speech_rate': await self._analyze_speech_rate(y, sr, sound),
                    'volume_level': await self._analyze_volume_level(y),
                    'voice_quality': await self._analyze_voice_quality(sound, y, sr),
                    'emotional_tone': await self._analyze_emotional_indicators(y, sr, sound),
                    'clarity_score': await self._analyze_speech_clarity(y, sr, sound),
                    'prosodic_features': await self._analyze_prosody(sound),
                    'spectral_features': await self._analyze_spectral_features(y, sr),
                    'voice_stability': await self._analyze_voice_stability(sound),
                    'medical_indicators': await self._detect_medical_voice_indicators(y, sr, sound)
                }
                
                return analysis_results
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Voice characteristics analysis error: {e}")
            # Return fallback analysis
            return await self._fallback_voice_analysis(audio_data)
    
    async def _analyze_speech_rate(self, y: np.ndarray, sr: int, sound) -> Dict[str, Any]:
        """Analyze speech rate and timing patterns"""
        try:
            # Detect syllables using intensity and spectral features
            # Method 1: Using intensity peaks
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")
            intensity_values = call(intensity, "List values", "all", "Hertz", "yes")
            
            # Find peaks in intensity (potential syllables)
            peaks, _ = scipy.signal.find_peaks(
                intensity_values, 
                height=np.mean(intensity_values) * 0.7,
                distance=int(sr * 0.1)  # Minimum 100ms between syllables
            )
            
            # Calculate speech rate
            duration = len(y) / sr
            syllables_per_second = len(peaks) / duration if duration > 0 else 0
            syllables_per_minute = syllables_per_second * 60
            
            # Classify speech rate
            if syllables_per_minute < 120:
                rate_category = "slow"
            elif syllables_per_minute < 180:
                rate_category = "normal"
            elif syllables_per_minute < 240:
                rate_category = "fast"
            else:
                rate_category = "very_fast"
            
            return {
                'syllables_per_minute': syllables_per_minute,
                'syllables_per_second': syllables_per_second,
                'category': rate_category,
                'total_syllables': len(peaks),
                'speech_duration': duration,
                'rhythm_regularity': np.std(np.diff(peaks)) if len(peaks) > 1 else 0
            }
            
        except Exception as e:
            logger.warning(f"Speech rate analysis failed: {e}")
            return {'category': 'unknown', 'syllables_per_minute': 0}
    
    async def _analyze_volume_level(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze volume level and dynamics"""
        try:
            # RMS energy analysis
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Statistics
            mean_volume = np.mean(rms_db)
            volume_variance = np.var(rms_db)
            dynamic_range = np.max(rms_db) - np.min(rms_db)
            
            # Categorize volume
            if mean_volume < -40:
                volume_category = "very_quiet"
            elif mean_volume < -25:
                volume_category = "quiet"
            elif mean_volume < -15:
                volume_category = "normal"
            elif mean_volume < -5:
                volume_category = "loud"
            else:
                volume_category = "very_loud"
            
            return {
                'category': volume_category,
                'mean_db': float(mean_volume),
                'variance_db': float(volume_variance),
                'dynamic_range_db': float(dynamic_range),
                'volume_stability': 1.0 / (1.0 + volume_variance) if volume_variance > 0 else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Volume analysis failed: {e}")
            return {'category': 'unknown', 'mean_db': 0}
    
    async def _analyze_voice_quality(self, sound, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze voice quality indicators"""
        try:
            # Extract fundamental frequency (F0)
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            # Jitter (frequency perturbation)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Shimmer (amplitude perturbation)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Harmonics-to-noise ratio
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_brightness = np.mean(spectral_centroids)
            
            # Voice quality assessment
            quality_score = self._calculate_voice_quality_score(jitter_local, shimmer_local, hnr)
            
            return {
                'jitter_percent': float(jitter_local * 100),
                'shimmer_percent': float(shimmer_local * 100),
                'hnr_db': float(hnr),
                'brightness_hz': float(mean_brightness),
                'quality_score': quality_score,
                'quality_category': self._categorize_voice_quality(quality_score),
                'roughness_indicator': jitter_local > 0.01 or shimmer_local > 0.1,
                'breathiness_indicator': hnr < 15.0
            }
            
        except Exception as e:
            logger.warning(f"Voice quality analysis failed: {e}")
            return {'quality_category': 'unknown', 'quality_score': 0.5}
    
    async def _analyze_emotional_indicators(self, y: np.ndarray, sr: int, sound) -> Dict[str, Any]:
        """Analyze emotional indicators from voice"""
        try:
            # Pitch analysis for emotion
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = call(pitch, "List values", "all", "Hertz", "yes")
            pitch_values = [p for p in pitch_values if p > 0]  # Remove unvoiced frames
            
            if len(pitch_values) == 0:
                return {'emotional_tone': 'uncertain', 'confidence': 0.0}
            
            # Pitch statistics
            mean_f0 = np.mean(pitch_values)
            f0_std = np.std(pitch_values)
            f0_range = np.max(pitch_values) - np.min(pitch_values)
            
            # Energy analysis
            rms = librosa.feature.rms(y=y)[0]
            energy_variance = np.var(rms)
            
            # Formant analysis
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Extract first two formants
            f1_values = []
            f2_values = []
            
            for i in range(1, int(call(formant, "Get number of frames")) + 1):
                f1 = call(formant, "Get value at time", 1, call(formant, "Get time from frame number", i), "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, call(formant, "Get time from frame number", i), "Hertz", "Linear")
                
                if not np.isnan(f1) and f1 > 0:
                    f1_values.append(f1)
                if not np.isnan(f2) and f2 > 0:
                    f2_values.append(f2)
            
            # Emotional state analysis
            emotional_features = {
                'mean_f0': mean_f0,
                'f0_std': f0_std,
                'f0_range': f0_range,
                'energy_variance': energy_variance,
                'f1_mean': np.mean(f1_values) if f1_values else 0,
                'f2_mean': np.mean(f2_values) if f2_values else 0
            }
            
            # Rule-based emotional classification
            emotion, confidence = self._classify_emotion(emotional_features)
            
            return {
                'emotional_tone': emotion,
                'confidence': confidence,
                'pitch_mean_hz': float(mean_f0),
                'pitch_variability': float(f0_std),
                'pitch_range_hz': float(f0_range),
                'energy_variability': float(energy_variance),
                'stress_indicators': {
                    'high_pitch': mean_f0 > 200,  # Hz
                    'pitch_instability': f0_std > 50,
                    'energy_fluctuation': energy_variance > 0.01
                }
            }
            
        except Exception as e:
            logger.warning(f"Emotional analysis failed: {e}")
            return {'emotional_tone': 'neutral', 'confidence': 0.5}
    
    async def _analyze_speech_clarity(self, y: np.ndarray, sr: int, sound) -> float:
        """Analyze speech clarity and articulation"""
        try:
            # Spectral clarity measures
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # High-frequency energy (indicates clear consonants)
            stft = librosa.stft(y)
            high_freq_energy = np.mean(np.abs(stft[int(len(stft) * 0.7):, :]))
            total_energy = np.mean(np.abs(stft))
            
            clarity_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Formant clarity (vowel definition)
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            formant_clarity = call(formant, "Get mean", 1, 0, 0, "Hertz")
            
            # Combine measures
            clarity_score = min(1.0, (clarity_ratio * 2 + 
                                    (formant_clarity / 1000 if not np.isnan(formant_clarity) else 0.5) + 
                                    np.mean(spectral_centroid) / 5000) / 3)
            
            return float(clarity_score)
            
        except Exception as e:
            logger.warning(f"Clarity analysis failed: {e}")
            return 0.5
    
    async def _analyze_prosody(self, sound) -> Dict[str, Any]:
        """Analyze prosodic features (intonation, stress patterns)"""
        try:
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")
            
            # Pitch contour analysis
            pitch_values = call(pitch, "List values", "all", "Hertz", "yes")
            pitch_values = [p for p in pitch_values if p > 0]
            
            # Intensity contour
            intensity_values = call(intensity, "List values", "all", "Hertz", "yes")
            
            if len(pitch_values) < 10 or len(intensity_values) < 10:
                return {'intonation_pattern': 'insufficient_data'}
            
            # Calculate prosodic measures
            pitch_slope = np.polyfit(range(len(pitch_values)), pitch_values, 1)[0]
            pitch_curvature = np.mean(np.diff(pitch_values, 2)) if len(pitch_values) > 2 else 0
            
            # Stress pattern detection
            stress_peaks, _ = scipy.signal.find_peaks(
                intensity_values, 
                height=np.mean(intensity_values) * 1.2,
                distance=int(len(intensity_values) * 0.05)
            )
            
            return {
                'pitch_slope': float(pitch_slope),
                'pitch_curvature': float(pitch_curvature),
                'stress_peaks': len(stress_peaks),
                'intonation_pattern': self._classify_intonation(pitch_slope, pitch_curvature),
                'prosodic_emphasis': len(stress_peaks) / (len(intensity_values) / 100) if len(intensity_values) > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}")
            return {'intonation_pattern': 'unknown'}
    
    async def _analyze_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze spectral characteristics"""
        try:
            # Extract various spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # MFCCs for voice characterization
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
                'spectral_complexity': float(np.std(spectral_centroids))
            }
            
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {'spectral_centroid_mean': 0}
    
    async def _analyze_voice_stability(self, sound) -> Dict[str, Any]:
        """Analyze voice stability and tremor indicators"""
        try:
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = call(pitch, "List values", "all", "Hertz", "yes")
            pitch_values = [p for p in pitch_values if p > 0]
            
            if len(pitch_values) < 10:
                return {'stability_score': 0.5, 'tremor_detected': False}
            
            # Calculate pitch stability
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            coefficient_of_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            # Detect potential tremor (regular oscillations)
            fft = np.fft.fft(pitch_values - np.mean(pitch_values))
            frequencies = np.fft.fftfreq(len(pitch_values))
            
            # Look for tremor frequency (typically 3-12 Hz)
            tremor_band = (frequencies >= 0.05) & (frequencies <= 0.2)  # Normalized frequency
            tremor_energy = np.sum(np.abs(fft[tremor_band]))
            total_energy = np.sum(np.abs(fft))
            
            tremor_ratio = tremor_energy / total_energy if total_energy > 0 else 0
            tremor_detected = tremor_ratio > 0.1
            
            stability_score = 1.0 / (1.0 + coefficient_of_variation * 10)
            
            return {
                'stability_score': float(stability_score),
                'coefficient_of_variation': float(coefficient_of_variation),
                'tremor_detected': tremor_detected,
                'tremor_ratio': float(tremor_ratio),
                'pitch_variability': float(pitch_std)
            }
            
        except Exception as e:
            logger.warning(f"Voice stability analysis failed: {e}")
            return {'stability_score': 0.5, 'tremor_detected': False}
    
    async def _detect_medical_voice_indicators(self, y: np.ndarray, sr: int, sound) -> Dict[str, Any]:
        """Detect voice indicators relevant to medical conditions"""
        try:
            # Respiratory analysis
            pauses = self._detect_pauses(y, sr)
            
            # Voice fatigue indicators
            energy_decay = self._analyze_energy_decay(y)
            
            # Articulation precision
            consonant_clarity = self._analyze_consonant_clarity(y, sr)
            
            # Vocal effort indicators
            vocal_effort = self._analyze_vocal_effort(sound)
            
            return {
                'respiratory_patterns': pauses,
                'voice_fatigue_indicators': energy_decay,
                'articulation_precision': consonant_clarity,
                'vocal_effort': vocal_effort,
                'medical_flags': {
                    'frequent_pauses': pauses['pause_frequency'] > 0.3,
                    'energy_decline': energy_decay['decay_rate'] > 0.1,
                    'reduced_clarity': consonant_clarity < 0.6,
                    'high_effort': vocal_effort > 0.7
                }
            }
            
        except Exception as e:
            logger.warning(f"Medical indicators analysis failed: {e}")
            return {'medical_flags': {}}
    
    # Helper methods
    def _calculate_voice_quality_score(self, jitter: float, shimmer: float, hnr: float) -> float:
        """Calculate overall voice quality score"""
        # Normalize and combine measures
        jitter_score = max(0, 1 - jitter * 100)  # Lower jitter is better
        shimmer_score = max(0, 1 - shimmer * 10)  # Lower shimmer is better
        hnr_score = min(1, hnr / 20)  # Higher HNR is better
        
        return (jitter_score + shimmer_score + hnr_score) / 3
    
    def _categorize_voice_quality(self, quality_score: float) -> str:
        """Categorize voice quality based on score"""
        if quality_score > 0.8:
            return "excellent"
        elif quality_score > 0.6:
            return "good"
        elif quality_score > 0.4:
            return "fair"
        elif quality_score > 0.2:
            return "poor"
        else:
            return "very_poor"
    
    def _classify_emotion(self, features: Dict[str, float]) -> tuple:
        """Rule-based emotion classification"""
        mean_f0 = features['mean_f0']
        f0_std = features['f0_std']
        energy_var = features['energy_variance']
        
        # Simple rule-based classification
        if mean_f0 > 250 and f0_std > 40:
            return "anxious", 0.7
        elif mean_f0 < 150 and energy_var < 0.005:
            return "depressed", 0.6
        elif f0_std > 60 and energy_var > 0.01:
            return "agitated", 0.8
        elif mean_f0 > 200 and energy_var > 0.008:
            return "stressed", 0.7
        else:
            return "neutral", 0.5
    
    def _classify_intonation(self, pitch_slope: float, pitch_curvature: float) -> str:
        """Classify intonation pattern"""
        if abs(pitch_slope) < 0.1:
            return "flat"
        elif pitch_slope > 0.5:
            return "rising"
        elif pitch_slope < -0.5:
            return "falling"
        elif abs(pitch_curvature) > 0.1:
            return "complex"
        else:
            return "normal"
    
    def _detect_pauses(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect and analyze pauses in speech"""
        # Simple energy-based pause detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(rms) * 0.1
        
        silent_frames = rms < threshold
        pauses = []
        
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * hop_length / sr
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pauses.append(pause_duration)
        
        return {
            'total_pauses': len(pauses),
            'pause_frequency': len(pauses) / (len(y) / sr) if len(y) > 0 else 0,
            'average_pause_duration': np.mean(pauses) if pauses else 0,
            'pause_durations': pauses
        }
    
    def _analyze_energy_decay(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze energy decay over time (voice fatigue indicator)"""
        # Split audio into segments
        segment_length = len(y) // 10  # 10 segments
        if segment_length < 1:
            return {'decay_rate': 0}
        
        segment_energies = []
        for i in range(10):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(y))
            segment = y[start:end]
            energy = np.mean(segment ** 2)
            segment_energies.append(energy)
        
        # Calculate decay rate
        if len(segment_energies) > 1:
            decay_rate = np.polyfit(range(len(segment_energies)), segment_energies, 1)[0]
        else:
            decay_rate = 0
        
        return {
            'decay_rate': float(abs(decay_rate)),
            'initial_energy': segment_energies[0] if segment_energies else 0,
            'final_energy': segment_energies[-1] if segment_energies else 0
        }
    
    def _analyze_consonant_clarity(self, y: np.ndarray, sr: int) -> float:
        """Analyze consonant clarity"""
        # High-frequency content analysis (consonants have more high-freq energy)
        stft = librosa.stft(y)
        high_freq_start = int(len(stft) * 0.6)  # Above 60% of Nyquist
        
        high_freq_energy = np.mean(np.abs(stft[high_freq_start:, :]))
        total_energy = np.mean(np.abs(stft))
        
        clarity_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        return min(1.0, clarity_ratio * 5)  # Scale and cap at 1.0
    
    def _analyze_vocal_effort(self, sound) -> float:
        """Analyze vocal effort indicators"""
        try:
            # Higher fundamental frequency can indicate effort/strain
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            pitch_values = call(pitch, "List values", "all", "Hertz", "yes")
            pitch_values = [p for p in pitch_values if p > 0]
            
            if not pitch_values:
                return 0.5
            
            mean_f0 = np.mean(pitch_values)
            # Higher pitch often indicates more vocal effort
            effort_score = min(1.0, (mean_f0 - 100) / 200) if mean_f0 > 100 else 0
            
            return max(0.0, effort_score)
            
        except:
            return 0.5
    
    async def _fallback_voice_analysis(self, audio_data: bytes) -> Dict[str, Any]:
        """Fallback analysis when libraries fail"""
        # Basic analysis without advanced libraries
        duration = len(audio_data) / (2 * 16000)  # Assume 16kHz, 16-bit
        
        return {
            'duration': duration,
            'speech_rate': {'category': 'unknown'},
            'volume_level': {'category': 'moderate'},
            'voice_quality': {'quality_category': 'unknown'},
            'emotional_tone': {'emotional_tone': 'neutral'},
            'clarity_score': 0.5,
            'prosodic_features': {'intonation_pattern': 'unknown'},
            'spectral_features': {'spectral_centroid_mean': 0},
            'voice_stability': {'stability_score': 0.5},
            'medical_indicators': {'medical_flags': {}}
        }
