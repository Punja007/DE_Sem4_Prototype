from youtube_api import search_youtube_videos
from captions import get_captions
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

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
    exit()

# Save transcript
with open("transcript_temp.txt", "w", encoding="utf-8") as f:
    f.write(captions)

# Function to chunk text
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(captions)
print(f"✂️ Split into {len(chunks)} chunks.")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
chunk_data = list(zip(chunks, embeddings))

# Convert embeddings to float32 for FAISS
embeddings_np = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)
print(f"🎯 Added {len(embeddings)} embeddings to FAISS index.")

# Save index + chunks
faiss.write_index(index, "faiss_index.index")
print("💾 FAISS index saved to 'faiss_index.index'")

np.save("chunks.npy", chunks)
with open("chunk_data.pkl", "wb") as f:
    pickle.dump(chunk_data, f)
print("🧠 Chunks and chunk data saved.")
