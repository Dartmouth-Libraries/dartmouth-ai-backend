from dartmouth_ai_backend.object_detection import ObjectDetector
from dartmouth_ai_backend.base.auth import get_jwt
from dartmouth_ai_backend.langchain_components import DartmouthChatModel
from dartmouth_ai_backend.langchain_components import DartmouthEmbeddings
from dartmouth_ai_backend.language_detection import LanguageDetector
from dartmouth_ai_backend.named_entity_recognition import NamedEntityRecognizer
from dartmouth_ai_backend.sentiment_analysis import SentimentAnalyzer
from dartmouth_ai_backend.speaker_diarization import SpeakerDiarizer
from dartmouth_ai_backend.speech_recognition import SpeechRecognizer

from dotenv import load_dotenv

from pathlib import Path


load_dotenv(Path(__file__).parent.parent / "secrets.env")


def test_auth():
    jwt = get_jwt()
    assert jwt


def test_citations():
    for obj in [
        ObjectDetector,
        LanguageDetector,
        NamedEntityRecognizer,
        SentimentAnalyzer,
        SpeechRecognizer,
        SpeakerDiarizer,
    ]:
        assert obj.how_to_cite()


def test_dartmouth_chat():
    llm = DartmouthChatModel()
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
    assert response.strip() == "OK"

    llm = DartmouthChatModel(model_name="codellama-13b-instruct-hf")
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
    assert response.strip() == "OK"

    llm = DartmouthChatModel(
        inference_server_url="https://ai-api.dartmouth.edu/tgi/codellama-13b-instruct-hf/",
    )
    print(llm.invoke("<s>[INST]Hello[/INST]"))


def test_dartmouth_embeddings():
    embeddings = DartmouthEmbeddings()
    result = embeddings.embed_query("Is there anybody out there?")
    assert result


def test_language_detection():
    with open(Path(__file__).parent.resolve() / "de.txt") as f:
        de_text = f.read()

    r = LanguageDetector().detect(de_text)
    assert r["language"], "de"

    with open(Path(__file__).parent.resolve() / "en.txt") as f:
        en_text = f.read()

    r = LanguageDetector().detect(en_text)
    assert r["language"], "en"


def test_named_entity_recognition():
    with open(Path(__file__).parent.resolve() / "wiki.txt") as f:
        raw_text = f.read()

    ner = NamedEntityRecognizer()
    result = ner.recognize(raw_text)
    assert result


def test_object_detection():
    result = ObjectDetector().detect(
        Path(__file__).parent.resolve() / "object_detection_sample.jpg"
    )
    assert result


def test_speaker_diarization():
    transcript = SpeechRecognizer(model="tiny", model_cache=".cache/").transcribe(
        str(Path(__file__).parent.resolve() / "speaker_diarization_sample.wav")
    )
    result = SpeakerDiarizer().diarize(
        str(Path(__file__).parent.resolve() / "speaker_diarization_sample.wav"),
        transcript=transcript,
    )
    assert result


def test_speech_recognition():
    result = SpeechRecognizer(model="tiny", model_cache=".cache/").transcribe(
        str(Path(__file__).parent.resolve() / "speech_recognition_sample.flac"),
        diarize=True,
    )

    result = SpeechRecognizer(model="medium", model_cache=".cache/").transcribe(
        str(Path(__file__).parent.resolve() / "speech_translation_sample.mp3"),
        task="translate",
    )
    assert result


if __name__ == "__main__":
    test_auth()
    test_dartmouth_chat()
    test_language_detection()
    test_named_entity_recognition()
    test_object_detection()
    test_speech_recognition()
    test_speaker_diarization()
    test_citations()
    test_dartmouth_embeddings()
