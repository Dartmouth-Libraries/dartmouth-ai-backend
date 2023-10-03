import whisper

class SpeechRecognizer:
    def __init__(self, model="large-v2"):
        self.__model = whisper.load_model(model)

    def transcribe(self, speech_file, task="transcribe", language=None) -> str:
        transcription = self.__model.transcribe(speech_file, task=task)

        return transcription
