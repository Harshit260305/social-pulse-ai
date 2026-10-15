import pandas as pd

from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_sentiment_pipeline():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME
    )

    return sentiment_pipeline

def predict_sentiment(text, sentiment_pipeline):

    # Step 1: Handle missing input
    if text is None or pd.isna(text):
        return {
            "label": None,
            "score": None,
            "error": "Empty text"
        }

    # Step 2: Convert non-string input to a string
    text = str(text)

    # Step 3: Handle empty text
    if text.strip() == "":
        return {
            "label": None,
            "score": None,
            "error": "Empty text"
        }

    # Step 4: Predict sentiment
    result = sentiment_pipeline(
        text,
        truncation=True,
        max_length=512
    )

    return result[0]

def analyze_sentiment(text, sentiment_pipeline):
    result = predict_sentiment(text, sentiment_pipeline)

    if result.get("error"):
        return {
            "sentiment": None,
            "confidence": None,
            "error": result["error"]
        }

    return {
        "sentiment": result["label"],
        "confidence": result["score"],
        "error": None
    }