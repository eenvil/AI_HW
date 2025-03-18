import requests
import csv
import time
import urllib.parse

def get_reviews(app_id, max_reviews=500):
    """
    Fetch reviews for a given Steam game (by app_id).
    The function collects reviews until it reaches max_reviews or no more reviews are available.
    """
    reviews = []
    cursor = '*'  # initial cursor value
    while len(reviews) < max_reviews:
        # URL-encode the cursor in case it contains special characters
        cursor_encoded = urllib.parse.quote(cursor)
        url = (f'https://store.steampowered.com/appreviews/{app_id}'
               f'?json=1&filter=all&language=english&cursor={cursor_encoded}'
               f'&review_type=all&purchase_type=all')
        try:
            response = requests.get(url)
            response.raise_for_status()  # raise error if status code is not 200
            data = response.json()
        except Exception as e:
            print(f"Error fetching reviews for app {app_id}: {e}")
            break

        # Check if reviews are available in the response
        new_reviews = data.get('reviews', [])
        if not new_reviews:
            print("No more reviews found.")
            break

        # Extract comment and sentiment (voted_up) from each review
        for review in new_reviews:
            review_text = review.get('review', '')
            voted_up = review.get('voted_up', False)  # True if positive, False otherwise
            reviews.append({
                'app_id': app_id,
                'review': review_text,
                'voted_up': voted_up
            })
            if len(reviews) >= max_reviews:
                break

        # Update cursor for the next page; if no new cursor, then exit the loop
        cursor = data.get('cursor')
        if not cursor:
            break

        # Pause briefly to avoid hitting rate limits
        time.sleep(1)

    return reviews

def main():
    # List of Steam app IDs for the games you want to collect reviews from.
    # For example: 620 (Portal 2), 440 (Team Fortress 2)...
    app_ids = [620, 440, 570, 730, 578080, 413150, 1091500, 1086940]
    all_reviews = []

    for app_id in app_ids:
        print(f"Fetching reviews for app {app_id}...")
        app_reviews = get_reviews(app_id, max_reviews=500)  # adjust max_reviews as needed
        print(f"Collected {len(app_reviews)} reviews for app {app_id}.")
        all_reviews.extend(app_reviews)

    # Save the collected reviews to a CSV file.
    csv_file = 'steam_reviews.csv'
    fieldnames = ['app_id', 'review', 'voted_up']
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_reviews)
        print(f"Saved a total of {len(all_reviews)} reviews to '{csv_file}'.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    main()
