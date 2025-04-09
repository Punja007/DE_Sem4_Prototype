import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube_videos(query, max_results=5):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )

    response = request.execute()

    results = []
    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        results.append({
            'title': title,
            'channel': channel,
            'url': video_url
        })

    return results
