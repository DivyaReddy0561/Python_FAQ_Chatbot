import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load dataset ---
# NOTE: Make sure the file path is correct on your system when you run this!
df = pd.read_csv(
    r"C:\Users\reddy\OneDrive\Desktop\SEM_07\NLP\NLP_Chatbot_Assignment\Python FAQ Dataset.csv",
    encoding="latin1"
)

# --- Clean dataset ---
df.columns = df.columns.str.strip()
df["Questions"] = df["Questions"].str.lower()
df.drop_duplicates(inplace=True)
df.dropna(subset=["Questions", "Answers"], inplace=True)

# --- Load pre-trained model ---
print("Loading model and encoding FAQ questions...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
question_embeddings = model.encode(df['Questions'].tolist())
print("Ready!")
# ----------------------------------------------------------------------

# --- Function to shorten answer for simple questions (Max 3 sentences) ---
def shorten_answer_for_simple_questions(user_question, answer):
    patterns = ["what is", "define", "difference between", "difference of", "tell me about"]
    
    # Flag to check if shortening was applied
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
    
    return final_answer, shortened # Return both the answer and the flag

# --- Function to format answer into readable paragraphs ---
def format_paragraphs(answer):
    sentences = re.split(r'\. |\n', answer)
    paragraph_list = []
    # Join sentences into paragraphs of 3 for long, complex answers
    for i in range(0, len(sentences), 3):
        paragraph = ". ".join(sentences[i:i+3]).strip()
        if paragraph:
            if not paragraph.endswith('.'):
                 paragraph += '.'
            paragraph_list.append(paragraph)
    return "\n\n".join(paragraph_list)

# --- Function to get best answer (MODIFIED) ---
def get_best_answer(user_question, df, question_embeddings, model):
    user_embedding = model.encode([user_question.lower()])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    
    best_match_index = similarities.argmax()
    max_similarity = similarities[0, best_match_index]
    
    # --- CONFIDENCE THRESHOLD LOGIC ---
    THRESHOLD = 0.30

    
    if max_similarity < THRESHOLD:
        return (
            f"I'm sorry, I couldn't find a relevant Python FAQ for that question. "
            f"(Confidence: {max_similarity:.2f} out of {THRESHOLD}). "
            "Please try asking a different question."
        )
    else:
        # High confidence match: Retrieve and format the answer
        answer = df.iloc[best_match_index]['Answers']
        
        # 1. Check for shortening first
        formatted_answer, was_shortened = shorten_answer_for_simple_questions(user_question, answer)
        
        # 2. Only apply paragraph formatting if the answer was NOT shortened
        if not was_shortened:
            formatted_answer = format_paragraphs(formatted_answer)
        
        # Return answer with confidence score
        return f"[Confidence: {max_similarity:.2f}]\n{formatted_answer}"

# --- Chat loop ---
while True:
    user_question = input("\nAsk a Python question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        break
    
    print("\nBot:", get_best_answer(user_question, df, question_embeddings, model))