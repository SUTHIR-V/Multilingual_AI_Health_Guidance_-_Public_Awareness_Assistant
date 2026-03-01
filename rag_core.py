import os
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
# -------------------- CONFIG -------------------- #

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

current_topic = None  # Advanced topic memory

groq_client = None

def initialize_groq(api_key):
    global groq_client
    groq_client = Groq(api_key=api_key)

# -------------------- LOAD RESOURCES -------------------- #

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        os.path.join(BASE_PATH, "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


vectorstore = load_vectorstore()


# -------------------- SYSTEM PROMPT -------------------- #

SYSTEM_PROMPT = """
You are a public health guidance assistant.

Use ONLY the provided context to answer the question.
Do NOT invent medical information.
Do NOT use outside knowledge.
If the information is not present in the context, clearly state:
"Not available in provided documents."

Formatting Rules:
- Answer strictly according to the user's question.
- Do NOT use a fixed template.
- Do NOT include sections that are not relevant to the question.
- If the user asks only about schemes, respond only about schemes.
- If the user asks about symptoms, respond only with symptoms.
- Keep the answer clear and structured when appropriate, but adapt naturally.

Always include a brief disclaimer at the end:
"This information is for educational purposes only and is not a substitute for professional medical advice."
"""


# -------------------- UNIQUE RETRIEVAL -------------------- #

def retrieve_unique_documents(query, k=5, final_k=3):
    docs = vectorstore.similarity_search(query, k=k)

    seen_sources = set()
    unique_docs = []

    for doc in docs:
        source = doc.metadata.get("source_file")

        if source not in seen_sources:
            unique_docs.append(doc)
            seen_sources.add(source)

        if len(unique_docs) >= final_k:
            break

    return unique_docs


# -------------------- TOPIC INFERENCE -------------------- #

def infer_topic_from_docs(docs):
    topic_count = {}

    for doc in docs:
        source = doc.metadata.get("source_file")
        if source:
            topic = source.replace(".pdf", "")
            topic_count[topic] = topic_count.get(topic, 0) + 1

    if topic_count:
        return max(topic_count, key=topic_count.get)

    return None


# -------------------- MAIN RAG FUNCTION -------------------- #

def ask_health_assistant(query):
    global current_topic

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

    # -------- First Retrieval Attempt -------- #
    docs = retrieve_unique_documents(translated_query)
    retrieved_topic = infer_topic_from_docs(docs)

    # -------- Detect Explicit Topic Mention -------- #
    query_lower = translated_query.lower()
    explicit_topic_mentioned = False

    if current_topic and current_topic.lower() in query_lower:
        explicit_topic_mentioned = True

    # -------- Smart Topic Override Logic -------- #
    if current_topic and not explicit_topic_mentioned:
        if retrieved_topic and retrieved_topic != current_topic:
            # Force topic continuity
            final_query = current_topic + " " + translated_query
            docs = retrieve_unique_documents(final_query)
            retrieved_topic = infer_topic_from_docs(docs)
        else:
            final_query = translated_query
    else:
        final_query = translated_query

    # -------- Update Current Topic -------- #
    if retrieved_topic:
        current_topic = retrieved_topic

    # -------- Build Context -------- #
    context = "\n\n".join([doc.page_content for doc in docs])

    message = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{final_query}"

    # -------- Generate Response -------- #
    response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{final_query}"}
    ],
    temperature=0.1,
    max_tokens=400
    )

    answer_en = response.choices[0].message.content
    # -------- Translate Back -------- #
    if user_lang != "en":
        final_answer = GoogleTranslator(
            source="en", target=user_lang
        ).translate(answer_en)
    else:
        final_answer = answer_en

    return final_answer
