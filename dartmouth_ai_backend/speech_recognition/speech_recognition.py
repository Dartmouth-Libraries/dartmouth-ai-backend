import whisper
import torch
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
        waveform, sample_rate = librosa.load(speech_file, mono=True, sr=16000)
        transcription = self.__model.transcribe(waveform, task=task, language=language)

        return transcription
