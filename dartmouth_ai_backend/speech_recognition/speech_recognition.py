from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf


class SpeechRecognizer:
    def __init__(self, model="large-v2"):
        model_name = f"openai/whisper-{model}"
        self.__processor = WhisperProcessor.from_pretrained(model_name)
        self.__model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.__model.config.forced_decoder_ids = None

    def transcribe(self, speech_file, skip_special_tokens=True) -> str:
        data, sampling_rate = sf.read(speech_file)
        input_features = self.__processor(
            data, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features

        predicted_ids = self.__model.generate(input_features)
        transcription = self.__processor.batch_decode(
            predicted_ids, skip_special_tokens=skip_special_tokens
        )
        return transcription
