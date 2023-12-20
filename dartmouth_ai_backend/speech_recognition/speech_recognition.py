import whisper
import torch
import torchaudio
import librosa


class SpeechRecognizer:
    def __init__(self, model="large-v2", device="cpu", model_cache=None):
        """Initializes the Speech Recognizer and loads the model

        Args:
            model (str, optional): Name of the model to load. Defaults to "large-v2".
            device (str, optional): Device to use for inference. Can be "cpu", "mps", or "cuda". Defaults to "cpu".
            model_cache (_type_, optional): Path to download the model file or load it from. Defaults to `~/.cache/whisper`.
        """
        self.__model = whisper.load_model(
            model, download_root=model_cache, device=torch.device(device)
        )

    def transcribe(self, speech_file, task="transcribe", language=None) -> str:
        # Librosa does not support loading MP3 from a BytesIO object, so go through torchaudio instead
        waveform, sample_rate = torchaudio.load(speech_file)
        # Mono conversion and resampling is more convenient in librosa
        waveform = librosa.to_mono(waveform.numpy())
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
        transcription = self.__model.transcribe(waveform, task=task, language=language)

        return transcription
