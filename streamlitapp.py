import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import numpy as np 
# ----------------------------
# Streamlit page config
st.set_page_config(page_title="Python FAQ Chatbot", page_icon="🐍", layout="wide")
st.title("Python FAQ Chatbot 🐍")
st.write("Welcome! Ask a question or click a history item on the left to see the details.")
# -----------------------------
# Force CPU (Good practice for deployment on shared hosting)
device = torch.device("cpu")
# -----------------------------
# Load dataset
@st.cache_data
def load_data():
    try:
        # NOTE: Ensure 'Python FAQ Dataset.csv' is accessible in the execution environment
        df = pd.read_csv("Python FAQ Dataset.csv", encoding="latin1")
        df.columns = df.columns.str.strip()
        df["Questions"] = df["Questions"].str.lower()
        df.drop_duplicates(inplace=True)
        df.dropna(subset=["Questions", "Answers"], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'Python FAQ Dataset.csv' not found. Please ensure the file is in the current directory.")
        st.stop() 
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
df = load_data()
# -----------------------------
# Helper function to shorten answer for simple questions (Max 3 sentences)
def shorten_answer_for_simple_questions(user_question, answer):
    patterns = ["what is", "define", "difference between", "difference of", "tell me about"]
    
    shortened = False 
    final_answer = answer
    if any(pat in user_question.lower() for pat in patterns):
        sentences = re.split(r'\. |\n', answer)
        
        # Set a clear limit for simple questions
        limit = min(3, len(sentences))
        final_answer = ". ".join(sentences[:limit]).strip()
        
        if final_answer and not final_answer.endswith('.'):
             final_answer += '.'
        # Add a note if truncation occurred
        if len(sentences) > limit:
             final_answer += " [...]"
        shortened = True  
    return final_answer, shortened

# --- Function to format answer into readable paragraphs (for complex answers) ---
def format_paragraphs(answer):
    sentences = re.split(r'\. |\n', answer)
    paragraph_list = []
    for i in range(0, len(sentences), 3):
        paragraph = ". ".join(sentences[i:i+3]).strip()
        if paragraph:
            if not paragraph.endswith('.'):
                 paragraph += '.'
            paragraph_list.append(paragraph)
    return "\n\n".join(paragraph_list)

# -----------------------------
# Load embeddings model (CPU only)
@st.cache_resource
def load_model():
    with st.spinner("Loading NLP Model..."):
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device=str(device))
model = load_model()

# Compute embeddings once and cache
@st.cache_data
def get_embeddings(questions, _model):
     with st.spinner("Encoding FAQ data for fast search..."):
        return _model.encode(questions)
question_embeddings = get_embeddings(df["Questions"].tolist(), model)

# -----------------------------
# Chatbot function (Final Logic)
def get_answer(user_question):
    query_embedding = model.encode([user_question.lower()])
    similarities = cosine_similarity(query_embedding, question_embeddings)

    best_idx = similarities.argmax()
    max_similarity_score = similarities[0, best_idx]
    # Confidence Threshold 
    THRESHOLD = 0.30 

    if max_similarity_score < THRESHOLD:
        return (
            f"I'm sorry, I couldn't find a highly relevant Python FAQ for that. "
            f"The confidence score ({max_similarity_score:.2f}) was below my threshold ({THRESHOLD:.2f}). "
            "Please try rephrasing your question or asking a different Python topic."
        )
    else:
        # Retrieve original answer for processing
        answer = df.iloc[best_idx]["Answers"]
        
        # Apply shortening logic first (for latest answer display)
        formatted_answer, was_shortened = shorten_answer_for_simple_questions(user_question, answer)
        
        # Only apply paragraph formatting if the answer was NOT shortened
        if not was_shortened:
            formatted_answer = format_paragraphs(formatted_answer)
        
        return f"**Confidence:** {max_similarity_score:.2f}\n\n{formatted_answer}"

# --- Retrieval function to get the FULL, UNTRUNCATED answer from the dataset ---
def get_full_original_answer(user_question):
    # Reruns the semantic search to find the best match index
    query_embedding = model.encode([user_question.lower()])
    similarities = cosine_similarity(query_embedding, question_embeddings)
    best_idx = similarities.argmax()
    
    # Retrieve the original full text
    original_answer_text = df.iloc[best_idx]["Answers"]
    
    # Apply full formatting (paragraphs) for readability, but NO shortening
    full_formatted_text = format_paragraphs(original_answer_text)
    
    return full_formatted_text
# -----------------------------
# Initialize chat history and new state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_response" not in st.session_state:
    st.session_state.latest_response = None
if "selected_message_index" not in st.session_state:
    st.session_state.selected_message_index = -1 
# --- CLEAR HISTORY FUNCTION ---
def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.latest_response = None 
    st.session_state.selected_message_index = -1 
    if 'user_input' in st.session_state:
        st.session_state.user_input = ""

# --- INPUT HANDLING FUNCTION (Handles the actual query logic) ---
def handle_input():
    if st.session_state.user_input and st.session_state.user_input.strip() != "":
        user_question = st.session_state.user_input
        
        with st.spinner("Searching for the best FAQ match..."):
            answer = get_answer(user_question) # This returns the SHORTENED version
        
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", answer))
        
        st.session_state.latest_response = answer
        st.session_state.selected_message_index = -1 
        
        st.session_state.user_input = "" 
        st.rerun()
# -----------------------------
# MAIN CHAT INTERFACE 
# -----------------------------
st.markdown("---")
col_input, col_clear = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Ask your Python question here:", 
        key="user_input", 
        on_change=handle_input,
        label_visibility="collapsed"
    )
with col_clear:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) 
    st.button("Clear History", on_click=clear_chat_history, help="Erase all conversation history.")

# -----------------------------
# MAIN FOCUS AREA (Latest Answer / Selected History) 
# -----------------------------
st.subheader("🤖 Current Answer Focus")
st.markdown("---")
q_index = st.session_state.selected_message_index
history_length = len(st.session_state.chat_history)

if q_index != -1 and q_index < history_length:
    # 1. Display Selected History Item (FIXED TO RETRIEVE FULL ANSWER)
    q_index = st.session_state.selected_message_index 
    # Retrieve the user's question from history
    user_q = st.session_state.chat_history[q_index][1]
  
    # Call the new function to get the FULL, UNTRUNCATED answer
    full_bot_a = get_full_original_answer(user_q)
    
    st.markdown(f"### 🧑 **Selected Query**")
    st.markdown(f"> {user_q}")
    st.divider()
    st.markdown(f"### 🤖 **Full Answer**") # Updated heading for clarity
    st.markdown(full_bot_a) # Display the FULL answer
elif st.session_state.latest_response:
    # 2. Display Latest Response (This still shows the SHORTENED version)
    # --- START OF CODE ADDITION FOR LATEST QUERY CONTEXT ---
    # Retrieve the latest user question from history
    if history_length >= 2:
        # The user's question is always the second to last entry in a completed Q&A pair
        latest_user_q = st.session_state.chat_history[history_length - 2][1] 
        
        # Display the question clearly
        st.markdown(f"### 🧑 **Your Query**") 
        st.markdown(f"> {latest_user_q}")
        st.divider()
    # --- END OF CODE ADDITION ---
    st.markdown(f"### 🤖 **Latest Answer**")
    st.markdown(st.session_state.latest_response)

else:
    st.info("Start by typing a question above. The answer will appear here.")
# -----------------------------
# CHAT HISTORY DISPLAY (SIDEBAR) 
# -----------------------------
st.sidebar.subheader("Conversation History")
st.sidebar.markdown("_Click any entry below to show the full conversation here._")
st.sidebar.markdown("---")
if st.session_state.chat_history:
    history_length = len(st.session_state.chat_history)
    start_index = history_length - (2 if history_length % 2 == 0 else 1)
    for i in range(start_index, -1, -2): 
        user_q = st.session_state.chat_history[i][1]
        
        if st.sidebar.button(f"🧑 Query: {user_q}", key=f"hist_{i}"):
            st.session_state.selected_message_index = i 
            st.session_state.latest_response = None 
            st.rerun()
else:
    st.sidebar.info("History is empty.")
