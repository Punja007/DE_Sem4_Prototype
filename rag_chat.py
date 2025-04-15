import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Load embedding model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")
chunks = np.load("chunks.npy", allow_pickle=True)

# 🔐 OpenRouter API Key
OPENROUTER_API_KEY = "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"

# 🧠 Message memory
conversation_history = [
    {"role": "system", "content": "You are a helpful, friendly tutor bot. Always answer in a clear and engaging way."}
]

def ask_openrouter(conversation_history):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": conversation_history,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("💥 OpenRouter messed up:", str(e))
        print("📦 Raw response:", response.text)
        return "Error"

print("🤖 Ready to answer your questions about the video!")

while True:
    query = input("\n💬 Ask something (or type 'exit'): ").strip()
    if query.lower() == "exit":
        print("👋 Bye Bye.")
        break

    # Search for relevant chunk from transcript
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    matched_chunk = chunks[I[0][0]]

    # Inject chunk context into memory for this query
    conversation_history.append({
        "role": "user",
        "content": f"""Context from YouTube transcript:
---
{matched_chunk}
---
Now answer this question: {query}"""
    })

    # Get response and store it
    answer = ask_openrouter(conversation_history)
    print("\n🧠 Answer:", answer)

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })
