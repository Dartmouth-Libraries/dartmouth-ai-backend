from langchain.llms import HuggingFaceTextGenInference

from dartmouth_ai_backend.base import auth

from typing import Callable


class DartmouthChatModel(HuggingFaceTextGenInference):
    """
    Extends the LangChain class HuggingFaceTextGenInference for more convenient
    interaction with Dartmouth's instance of Text Generation Inference
    """

    authenticator: Callable = None
    dartmouth_api_key: str = None

    def __init__(
        self,
        *args,
        dartmouth_api_key: str = None,
        model_name="llama-2-13b-chat-hf",
        authenticator: Callable = None,
        jwt_url: str = None,
        inference_server_url: str = None,
        **kwargs,
    ):
        """
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
            model_name (str, optional): Name of the model to use. Defaults to "llama-2-13b-chat-hf".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
            inference_server_url (str, optional): URL pointing to an inference endpoint. Defaults to 'https://ai-api.dartmouth.edu/tgi/{model_name}/generate_stream'.
        """
        if inference_server_url:
            kwargs["inference_server_url"] = inference_server_url
        else:
            kwargs["inference_server_url"] = (
                f"https://ai-api.dartmouth.edu/tgi/{model_name}/generate_stream"
            )

        kwargs["streaming"] = True
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)

    def authenticate(self, jwt_url=None):
        if self.authenticator:
            jwt = self.authenticator()
        else:
            jwt = auth.get_jwt(
                dartmouth_api_key=self.dartmouth_api_key, jwt_url=jwt_url
            )
        self.client.headers = {"Authorization": f"Bearer {jwt}"}

    def predict(self, text: str, **kwargs) -> str:
        """Predict a response to a query.

        Args:
            text (str): The initial text, i.e., user query. If not properly formatted, it is wrapped in Llama-compatible tags:
            '<s>[INST]USER PROMPT GOES HERE[/INST]'

        Returns:
            str: The predicted continuation, i.e. model response.
        """
        if not text.startswith("<s>[INST]"):
            text = f"<s>[INST]{text}[/INST]"
        try:
            return super().predict(text, **kwargs)
        except KeyError:
            self.authenticate()
            return super().predict(text, **kwargs)
