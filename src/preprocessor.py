import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    
    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def preprocessed_dataframe(df):
    df["full_text"] = df["full_text"].apply(clean_text)
    return df


sample = "  Python <b>is amazing</b>!!! https://example.com  "

print(clean_text(sample))