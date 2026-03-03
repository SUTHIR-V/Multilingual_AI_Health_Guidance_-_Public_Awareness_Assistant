import os
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from huggingface_hub import login

# -------------------- CONFIG -------------------- #

BASE_PATH = "."
groq_client = None


def initialize_groq(api_key):
    global groq_client
    groq_client = Groq(api_key=api_key)


# -------------------- HF LOGIN -------------------- #

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# -------------------- LOAD VECTORSTORE -------------------- #

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        f"{BASE_PATH}/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()


# -------------------- SYSTEM PROMPT -------------------- #

SYSTEM_PROMPT = """
You are a public health guidance assistant.

Use ONLY the provided context to answer the question.
Do NOT invent medical information.
Do NOT use outside knowledge.
If the information is not present in the context, clearly state:
"Not available in provided documents."

Answer strictly according to the user's question.
Always include a short disclaimer at the end:
"This information is for educational purposes only and is not a substitute for professional medical advice."
"""


# -------------------- QUERY HELPERS -------------------- #

def analyze_query(query):
    q = query.lower()

    if any(word in q for word in ["symptom", "sign"]):
        return "symptoms"
    if any(word in q for word in ["prevent", "prevention"]):
        return "prevention"
    if any(word in q for word in ["scheme", "government", "programme", "program"]):
        return "scheme"

    return "general"


def detect_explicit_topic(query):
    q = query.lower()
    docs = vectorstore.similarity_search(query, k=15)

    for doc in docs:
        source = doc.metadata.get("source_file", "")
        topic = source.replace(".pdf", "").lower()
        if topic and topic in q:
            return topic

    return None


def is_ambiguous_query(query):
    ambiguous_terms = ["it", "its", "that", "those", "them"]
    q = query.lower()
    return any(
        q.strip().startswith(term) or f" {term} " in q
        for term in ambiguous_terms
    )


# -------------------- STRICT RETRIEVAL -------------------- #

def retrieve_documents(query, current_topic, k=10, final_k=3):

    intent = analyze_query(query)
    explicit_topic = detect_explicit_topic(query)
    ambiguous = is_ambiguous_query(query)

    # -------- Determine Topic -------- #
    if explicit_topic:
        topic = explicit_topic
    elif ambiguous and current_topic:
        topic = current_topic
    else:
        topic = None

    # -------- Force topic into embedding query -------- #
    if topic:
        search_query = f"{topic} {query}"
    else:
        search_query = query

    docs = vectorstore.similarity_search(search_query, k=k)

    filtered = []

    for doc in docs:
        source_file = doc.metadata.get("source_file", "").lower()
        category = doc.metadata.get("category", "").lower()

        # Strict exact match
        if topic:
            if source_file != f"{topic}.pdf":
                continue

        # Intent filtering
        if intent == "scheme" and category != "schemes":
            continue

        if intent in ["symptoms", "prevention"] and category != "diseases":
            continue

        filtered.append(doc)

    return filtered[:final_k], topic


# -------------------- MAIN RAG FUNCTION -------------------- #

def ask_health_assistant(query, current_topic=None):

    if not groq_client:
        raise ValueError("Groq client not initialized. Call initialize_groq().")

    # -------- Language Detection -------- #
    try:
        user_lang = detect(query)
    except:
        user_lang = "en"

    # -------- Translate to English -------- #
    if user_lang != "en":
        translated_query = GoogleTranslator(
            source=user_lang, target="en"
        ).translate(query)
    else:
        translated_query = query

    # -------- Retrieve -------- #
    docs, updated_topic = retrieve_documents(
        translated_query,
        current_topic
    )

    # Update topic if explicitly found
    explicit_topic = detect_explicit_topic(translated_query)
    if explicit_topic:
        updated_topic = explicit_topic

    # -------- If No Docs -------- #
    if not docs:
        answer_en = (
            "Not available in provided documents.\n\n"
            "This information is for educational purposes only and "
            "is not a substitute for professional medical advice."
        )
    else:
        context = "\n\n".join([doc.page_content[:1200] for doc in docs])
        context = context[:3500]

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{translated_query}"}
            ],
            temperature=0.1,
            max_tokens=400,
        )

        answer_en = response.choices[0].message.content

    # -------- Translate Back -------- #
    if user_lang != "en":
        final_answer = GoogleTranslator(
            source="en", target=user_lang
        ).translate(answer_en)
    else:
        final_answer = answer_en

    return final_answer, updated_topic
