import os
import pandas as pd


LEADERBOARD_FILE = "reports/leaderboard.csv"


def update_leaderboard(exp_id, accuracy, notes=""):
    row = {
        "experiment": exp_id,
        "accuracy": accuracy,
        "notes": notes,
    }

    os.makedirs("reports", exist_ok=True)

    if os.path.exists(LEADERBOARD_FILE):
        df = pd.read_csv(LEADERBOARD_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.sort_values(by="accuracy", ascending=False)
    df.to_csv(LEADERBOARD_FILE, index=False)

    return df.head(10)
