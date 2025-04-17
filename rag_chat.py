import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Load embedding model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")
chunks = np.load("chunks.npy", allow_pickle=True)

# 🔐 OpenRouter API Key
url = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"

# 🧠 Message memory
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a highly intelligent and helpful AI tutor bot. "
            "You read YouTube video transcripts and answer questions from users clearly, concisely, and informatively. "
            "You remember everything the user says during the session and can respond to both context-based and general questions. "
            "Always prioritize answering the user's intent. If the query is unrelated to the transcript, still respond helpfully."
        )
    }
]

def ask_openrouter(messages):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("\n💥 OpenRouter messed up:", str(e))
        print("\n📦 Raw response:", response.text)
        return "Sorry, something went wrong."

print("🤖 Ready to answer your questions about the video!")

while True:
    query = input("\n💬 Ask something (or type 'exit'): ").strip()
    if query.lower() == "exit":
        print("👋 Bye Bye.")
        break

    # Add user message first to maintain order in conversation
    conversation_history.append({"role": "user", "content": query})

    # Find the best matching chunk
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    matched_chunk = chunks[I[0][0]]

    # Add context if it makes sense
    context_note = f"Relevant video context: {matched_chunk}\n"
    conversation_history.append({
        "role": "user",
        "content": context_note + "Now answer based on above context and our chat history."
    })

    # Get response and print
    answer = ask_openrouter(conversation_history)
    print("\n🧠 Answer:", answer)

    # Store the assistant response
    conversation_history.append({"role": "assistant", "content": answer})