import whisper
import torch
import torchaudio
import librosa

from ..speaker_diarization import SpeakerDiarizer

from typing import Union, BinaryIO, Optional
from os import PathLike


class SpeechRecognizer:
    def __init__(self, model="large-v2", device="cpu", model_cache=None, diarize=False):
        """Initializes the Speech Recognizer and loads the model

        Args:
            model (str, optional): Name of the model to load. Defaults to "large-v2".
            device (str, optional): Device to use for inference. Can be "cpu", "mps", or "cuda". Defaults to "cpu".
            model_cache (str, optional): Path to download the model file or load it from. Defaults to `~/.cache/whisper`.
        """
        self.__model = whisper.load_model(
            model, download_root=model_cache, device=torch.device(device)
        )
        self.model_cache = model_cache
        self.device = device

    def transcribe(
        self,
        speech_file: Union[BinaryIO, str, PathLike],
        task: str = "transcribe",
        language: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> dict[str, str | list]:
        """Transcribes a speech file to text with optional labeling of the speaker ID.

        Args:
            speech_file (path-like object or file-like object): Speech file to process.
            task (str, optional): Task to perform ("transcribe" or "translate"). Defaults to "transcribe".
            language (str, optional): Language of the speech file. Defaults to None.
            diarize (bool, optional): Run speaker diarization. Defaults to False.
            num_speakers (int, optional): Number of speakers in the speech file. Defaults to None.
            min_speakers (int, optional): Minimum number of speakers in the speech file. Defaults to None.
            max_speakers (int, optional): Maximum number of speakers in the speech file. Defaults to None.

        Returns:
            dict[str, str | list]: A dictionary containing the resulting text ("text") and segment-level details ("segments"), and the spoken language ("language"), which is detected when "language" is None.
        """
        # Librosa does not support loading MP3 from a BytesIO object, so go through torchaudio instead
        waveform, sample_rate = torchaudio.load(speech_file)
        # Mono conversion and resampling is more convenient in librosa
        waveform = librosa.to_mono(waveform.numpy())
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
        transcription = self.__model.transcribe(waveform, task=task, language=language)

        if diarize:
            transcription = SpeakerDiarizer(
                model_cache=self.model_cache, device=self.device
            ).diarize(
                speech_file=speech_file,
                transcript=transcription,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        return transcription

    @staticmethod
    def how_to_cite(format="bibtex") -> str:
        if format != "bibtex":
            return NotImplemented
        return """@article{radford2022whisper,
    doi = {10.48550/ARXIV.2212.04356},
    url = {https://arxiv.org/abs/2212.04356},
    author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
    title = {Robust Speech Recognition via Large-Scale Weak Supervision},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
}"""
