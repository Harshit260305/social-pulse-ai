import requests
import pandas as pd


def fetch_raw_data(subreddit, limit):

    url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    params = {
        "subreddit": subreddit,
        "limit": limit,
        "fields": "id,title,selftext,score,num_comments,created_utc,subreddit"
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    posts = data.get("data") or []

    df = pd.DataFrame(posts)

    return df


def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = fetch_raw_data("datascience", 100)
    save_data(df, "data/raw_posts_large.csv")
