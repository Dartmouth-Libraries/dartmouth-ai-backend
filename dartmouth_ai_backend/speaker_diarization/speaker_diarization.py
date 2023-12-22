import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
import torchaudio

import logging
import os


class SpeakerDiarizer:
    def __init__(
        self,
        pipeline="pyannote/speaker-diarization-3.1",
        use_auth_token=None,
        device="cpu",
        model_cache=None,
    ):
        """Initializes the Speaker Diarizer and loads the pipeline

        Args:
            pipeline (str, optional): Name of the pipeline to load. Defaults to "pyannote/speaker-diarization-3.1".
            device (str, optional): Device to use for inference. Can be "cpu", "mps", or "cuda". Defaults to "cpu".
            model_cache (_type_, optional): Path to download the model file or load it from. Defaults to `.cache/hf/hub`.
        """

        if use_auth_token is None:
            use_auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

        if model_cache is None:
            model_cache = ".cache/hf/hub"
        logging.info("Loading diarization pipeline")
        self.__pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token,
            cache_dir=model_cache,
        )

    def diarize(
        self,
        speech_file,
        transcript=None,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ) -> pd.DataFrame:
        logging.info("Loading audio")
        waveform, sample_rate = torchaudio.load(speech_file)
        logging.info("Running diarization pipeline")
        diarization = self.__pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarization = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarization["start"] = diarization["segment"].apply(lambda x: x.start)
        diarization["end"] = diarization["segment"].apply(lambda x: x.end)

        if transcript is not None:
            transcript = self._assign_word_speakers(diarization, transcript)
            return transcript
        return diarization

    @staticmethod
    def how_to_cite(format="bibtex") -> str:
        if format != "bibtex":
            return NotImplemented
        return """@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}"""

    @staticmethod
    def _assign_word_speakers(
        diarize_df: pd.DataFrame, transcript_result: dict[str, str | list]
    ) -> list:
        """Assigns a speaker ID to each segment in a transcript

        This function is largely identical with whisperx' implementation (https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py).

        Args:
            diarize_df (pd.DataFrame): A pandas dataframe containing diarization information
            transcript_result (dict[str, str  |  list]): A transcript in the format produced by `SpeechRecognizer`

        Returns:
            list: A list of segments with assigned speaker IDs
        """
        transcript_segments = transcript_result["segments"]
        for seg in transcript_segments:
            # assign speaker to segment (if any)
            diarize_df["intersection"] = np.minimum(
                diarize_df["end"], seg["end"]
            ) - np.maximum(diarize_df["start"], seg["start"])
            diarize_df["union"] = np.maximum(
                diarize_df["end"], seg["end"]
            ) - np.minimum(diarize_df["start"], seg["start"])
            # remove no hit
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
            if len(dia_tmp) > 0:
                # sum over speakers
                speaker = (
                    dia_tmp.groupby("speaker")["intersection"]
                    .sum()
                    .sort_values(ascending=False)
                    .index[0]
                )
                seg["speaker"] = speaker

            # assign speaker to words
            if "words" in seg:
                for word in seg["words"]:
                    if "start" in word:
                        diarize_df["intersection"] = np.minimum(
                            diarize_df["end"], word["end"]
                        ) - np.maximum(diarize_df["start"], word["start"])
                        diarize_df["union"] = np.maximum(
                            diarize_df["end"], word["end"]
                        ) - np.minimum(diarize_df["start"], word["start"])
                        # remove no hit
                        dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                        if len(dia_tmp) > 0:
                            # sum over speakers
                            speaker = (
                                dia_tmp.groupby("speaker")["intersection"]
                                .sum()
                                .sort_values(ascending=False)
                                .index[0]
                            )
                            word["speaker"] = speaker

        return transcript_result
