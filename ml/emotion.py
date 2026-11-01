from transformers import pipeline
import pandas as pd
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

def load_emotion_pipeline():
    emotion_pipeline = pipeline(
        "text-classification",
        model = MODEL_NAME
    )

    return emotion_pipeline

def predict_emotion(text, emotion_pipeline):

    # step 1: Handle missing input
    if text is None or pd.isna(text):
        return{
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

    # Step 4: Predict emotion
    result = emotion_pipeline(
        text,
        truncation=True,
        max_length=512

    )
    return result[0]

def analyze_emotion(text, emotion_pipeline):
    result = predict_emotion(text, emotion_pipeline)

    if result.get("error"):
        return{
            "emotion": None,
            "confidence": None,
            "error": result["error"]
        }
    return{
        "emotion": result["label"],
        "confidence": result["score"],
        "error": None
    }