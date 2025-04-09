from youtube_api import search_youtube_videos
from captions import get_captions

choice = input("Do you have a specific YouTube video URL? (yes/no): ").strip().lower()

if choice == "yes":
    video_url = input("Paste your YouTube video URL: ").strip()
else:
    query = input("Enter a topic to learn: ")
    videos = search_youtube_videos(query)

    print("\n📚 Top 5 videos found:")
    for i, vid in enumerate(videos):
        print(f"{i+1}. {vid['title']} - by {vid['channel']}")

    selected = int(input("\nPick a video (1-5): "))
    selected_video = videos[selected - 1]
    video_url = selected_video['url']

    print(f"\n🎯 You selected: {selected_video['title']}")
    print(f"📺 URL: {video_url}")

print("\n⏬ Fetching Captions...")
captions = get_captions(video_url)

if "[Error getting captions]" in captions:
    print("\n🚫 Captions for the video are not available. Can't use this video. Pick another one!")
else:
    print("\n📜 Full Captions:\n")
    print(captions)
