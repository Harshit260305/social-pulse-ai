import pandas as pd
import re

def clean_text(text):
    if text is None:
        return ""
    
    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def preprocessed_dataframe(df):
    df["full_text"] = df["full_text"].apply(clean_text)
    return df
