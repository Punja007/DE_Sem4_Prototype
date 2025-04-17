import streamlit as st
from youtube_api import search_youtube_videos
from captions import get_captions
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
OPENROUTER_API_KEY = "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"  # Replace with your key

# Message history for conversation context
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are a helpful tutor bot."}]
    st.session_state.index = None
    st.session_state.chunks = []

def get_video_id_from_link(link):
    if "v=" in link:
        return link.split("v=")[1].split("&")[0]
    elif "youtu.be/" in link:
        return link.split("youtu.be/")[1]
    return None

def embed_and_index_chunks(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def ask_openrouter(messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "mistralai/mistral-7b-instruct", "messages": messages, "temperature": 0.7}
    response = requests.post(url, headers=headers, json=payload)
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return "⚠️ Something went wrong with OpenRouter."

def fetch_transcript_and_prepare_memory(video_url):
    video_id = get_video_id_from_link(video_url)
    if video_id:
        transcript = get_captions(video_url)
        if "Error" in transcript:
            return None, transcript
        chunks = [transcript[i:i+500] for i in range(0, len(transcript), 500)]
        index, chunks = embed_and_index_chunks(chunks)
        return index, chunks
    return None, "Invalid YouTube URL"

# ------------------------ UI Starts Here ------------------------

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🎥 YouTube Transcript Chatbot")

if st.session_state.index is None:
    input_type = st.radio("Do you have a YouTube link?", ("Yes", "No"))

    if input_type == "Yes":
        url = st.text_input("Paste your YouTube link:")
        if url:
            with st.spinner("Fetching transcript and preparing memory..."):
                index, chunks = fetch_transcript_and_prepare_memory(url)
                if index:
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.success("Ready to chat!")
                else:
                    st.error(f"❌ {chunks}")
    else:
        topic = st.text_input("Enter video topic:")
        if topic:
            results = search_youtube_videos(topic)
            for i, r in enumerate(results):
                st.write(f"{i+1}. [{r['title']}]({r['url']}) — *{r['channel']}*")
            choice = st.number_input("Choose video (1-5):", min_value=1, max_value=5, step=1)
            if st.button("Load selected video"):
                video_url = results[choice - 1]['url']
                with st.spinner("Fetching transcript and preparing memory..."):
                    index, chunks = fetch_transcript_and_prepare_memory(video_url)
                    if index:
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.success("Ready to chat!")
                    else:
                        st.error(f"❌ {chunks}")

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
