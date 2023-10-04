import whisper
import librosa


class SpeechRecognizer:
    def __init__(self, model="large-v2", model_cache=None):
        self.__model = whisper.load_model(model, download_root=model_cache)

    def transcribe(self, speech_file, task="transcribe", language=None) -> str:
        waveform, sample_rate = librosa.load(speech_file, mono=True, sr=16000)
        transcription = self.__model.transcribe(waveform, task=task, language=language)

        return transcription
