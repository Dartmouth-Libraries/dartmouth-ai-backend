# Dartmouth AI backend

This is a collection of simple tools to run basic machine learning tasks.

## Contents

- Language Detection
- Named Entity Recognition
- Object Detection
- Sentiment Analysis
- Speaker Diarization
- Speech Recognition
- Speech Translation
- A `LangChain`-compatible object to intarface Dartmouth-hosted Large Language Models

## Getting Started
1. Clone the repository 
2. Install from the repo root:
```
pip install .
```

## Running the tests
1. In the repo root, create a file called `secrets.env` with the following contents:
```
DARTMOUTH_API_KEY = <your_key_here>
```
2. Run the test script `test/test.py`
