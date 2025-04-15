# ✅ streamlit_chatbot.py
# Streamlit chatbot that handles video link or topic input in one flow

import streamlit as st
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch

model = SentenceTransformer('all-MiniLM-L6-v2')
OPENROUTER_API_KEY = "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"  # Replace with your key
conversation_history = []

def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    chunks = []
    temp_chunk = ""
    for entry in transcript:
        if len(temp_chunk) + len(entry['text']) < 500:
            temp_chunk += " " + entry['text']
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = entry['text']
    chunks.append(temp_chunk.strip())
    return chunks

def embed_and_index_chunks(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def ask_openrouter(messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
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
        return "⚠️ Something went wrong with OpenRouter."

def get_video_id_from_link(link):
    if "v=" in link:
        return link.split("v=")[1].split("&")[0]
    elif "youtu.be/" in link:
        return link.split("youtu.be/")[1]
    return None

def search_youtube_videos(query):
    results = YoutubeSearch(query, max_results=5).to_dict()
    return results

# ------------------------ UI Starts Here ------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🎥 YouTube Transcript Chatbot")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.history = [
        {"role": "system", "content": (
            "You are a helpful tutor chatbot that reads YouTube transcripts and answers user questions."
            " You remember previous messages and answer helpfully even if the question is off-topic."
        )}
    ]

if st.session_state.index is None:
    input_type = st.radio("Do you have a YouTube link?", ("Yes", "No"))

    if input_type == "Yes":
        url = st.text_input("Paste your YouTube link:")
        if url:
            video_id = get_video_id_from_link(url)
            if video_id:
                with st.spinner("Fetching transcript and preparing memory..."):
                    chunks = fetch_transcript(video_id)
                    index, chunks = embed_and_index_chunks(chunks)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.success("Ready to chat!")
            else:
                st.error("❌ Invalid YouTube link")

    else:
        topic = st.text_input("Enter video topic:")
        if topic:
            results = search_youtube_videos(topic)
            for i, r in enumerate(results):
                st.write(f"{i+1}. [{r['title']}]('https://www.youtube.com{r['url_suffix']}')")
            choice = st.number_input("Choose video (1-5):", min_value=1, max_value=5, step=1)
            if st.button("Load selected video"):
                video_id = results[choice - 1]['url_suffix'].split('v=')[1]
                with st.spinner("Fetching transcript and preparing memory..."):
                    chunks = fetch_transcript(video_id)
                    index, chunks = embed_and_index_chunks(chunks)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.success("Ready to chat!")

# ------------------------ Chat UI ------------------------

if st.session_state.index:
    st.header("💬 Chat")

    for msg in st.session_state.history:
        if msg['role'] in ['user', 'assistant']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

    user_input = st.chat_input("Ask me anything from the video!")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        query_embedding = model.encode([user_input])
        D, I = st.session_state.index.search(query_embedding, k=1)
        matched_chunk = st.session_state.chunks[I[0][0]]

        st.session_state.history.append({
            "role": "user",
            "content": f"Relevant video context: {matched_chunk}\nNow answer based on above context and our chat history."
        })

        answer = ask_openrouter(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": answer})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            st.write(answer)
