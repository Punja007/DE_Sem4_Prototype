from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(youtube_url):
    # extracts video ID from full YouTube URL
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    return None

def get_captions(video_url):
    try:
        video_id = get_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        return f"[Error getting captions]: {str(e)}"