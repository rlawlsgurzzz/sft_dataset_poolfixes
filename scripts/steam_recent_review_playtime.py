import csv
import statistics
import time
import requests

GAMES = {
    "Risk of Rain 2": 632360,
    "Hades": 1145360,
    "Hades II": 1145350,
    "Dead Cells": 588650,
    "Darkest Dungeon": 262060,
    "Rogue Legacy 2": 1253920,
}

BASE_URL = "https://store.steampowered.com/appreviews/{appid}"

def fetch_recent_reviews(appid: int, num_per_page: int = 100) -> list[dict]:
    params = {
        "json": 1,
        "filter": "recent",
        "language": "all",
        "purchase_type": "all",
        "num_per_page": num_per_page,
    }

    response = requests.get(BASE_URL.format(appid=appid), params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    if data.get("success") != 1:
        raise RuntimeError(f"Steam API failed for appid={appid}: {data}")

    return data.get("reviews", [])

def summarize_playtime(reviews: list[dict]) -> dict:
    values_minutes = []

    for review in reviews:
        author = review.get("author", {})
        value = author.get("playtime_last_two_weeks")

        if isinstance(value, int) and value >= 0:
            values_minutes.append(value)

    if not values_minutes:
        return {
            "sample_size": 0,
            "avg_2w_hours": None,
            "median_2w_hours": None,
            "avg_month_hours": None,
            "median_month_hours": None,
        }

    avg_minutes = statistics.mean(values_minutes)
    median_minutes = statistics.median(values_minutes)

    avg_2w_hours = avg_minutes / 60
    median_2w_hours = median_minutes / 60

    monthly_multiplier = 30 / 14

    return {
        "sample_size": len(values_minutes),
        "avg_2w_hours": avg_2w_hours,
        "median_2w_hours": median_2w_hours,
        "avg_month_hours": avg_2w_hours * monthly_multiplier,
        "median_month_hours": median_2w_hours * monthly_multiplier,
    }

def main():
    rows = []

    for game_name, appid in GAMES.items():
        print(f"Fetching {game_name} ({appid})...")

        reviews = fetch_recent_reviews(appid)
        summary = summarize_playtime(reviews)

        rows.append({
            "game": game_name,
            "appid": appid,
            "sample_size": summary["sample_size"],
            "avg_2w_hours": summary["avg_2w_hours"],
            "median_2w_hours": summary["median_2w_hours"],
            "avg_month_hours": summary["avg_month_hours"],
            "median_month_hours": summary["median_month_hours"],
            "api_url": BASE_URL.format(appid=appid),
        })

        time.sleep(1)

    with open("steam_recent_review_playtime.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "game",
            "appid",
            "sample_size",
            "avg_2w_hours",
            "median_2w_hours",
            "avg_month_hours",
            "median_month_hours",
            "api_url",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved: steam_recent_review_playtime.csv")

if __name__ == "__main__":
    main()