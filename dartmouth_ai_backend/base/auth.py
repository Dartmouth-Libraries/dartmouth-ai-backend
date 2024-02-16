"""Authentication and token management"""

import requests

import os

env_names = {"dartmouth_api_key": "DARTMOUTH_API_KEY"}


def get_jwt(dartmouth_api_key=None, jwt_url=None) -> str | None:
    if jwt_url is None:
        jwt_url = "https://api.dartmouth.edu/api/jwt"
    if dartmouth_api_key is None:
        dartmouth_api_key = os.getenv(env_names["dartmouth_api_key"])
    if dartmouth_api_key:
        r = requests.post(
            url=jwt_url,
            headers={"Authorization": dartmouth_api_key},
        )
        jwt = r.json()
        return jwt["jwt"]
    raise ValueError(
        f"Dartmouth API key not provided as argument or defined as environment variable {env_names['dartmouth_api_key']}."
    )
