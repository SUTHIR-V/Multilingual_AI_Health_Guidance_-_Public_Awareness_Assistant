import os
import io
import streamlit as st
from rag_core import ask_health_assistant, initialize_groq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------- PAGE CONFIG -------------------- #

st.set_page_config(page_title="AI Health Assistant", layout="centered")

# -------------------- INITIALIZE COHERE -------------------- #

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set")
    st.stop()

initialize_groq(GROQ_API_KEY)

# -------------------- TITLE -------------------- #

st.title("🌍 Multilingual AI Health Guidance Assistant")

# -------------------- SESSION STATE -------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY CHAT -------------------- #

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- CHAT INPUT -------------------- #

user_input = st.chat_input("Ask your health-related question...")

if user_input:

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_health_assistant(user_input)

        st.markdown(answer)

    # Store assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# -------------------- EXPORT CHAT TO PDF -------------------- #

def generate_chat_pdf(messages):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    normal_style = styles["Normal"]

    elements.append(Paragraph("AI Health Guidance Chat Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        text = f"<b>{role}:</b> {msg['content']}"
        elements.append(Paragraph(text, normal_style))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(PageBreak())

    elements.append(Paragraph(
        "Disclaimer: This information is for educational purposes only and "
        "is not a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider.",
        normal_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


if st.session_state.messages:
    pdf_buffer = generate_chat_pdf(st.session_state.messages)

    st.download_button(
        label="📄 Export Chat as PDF",
        data=pdf_buffer,
        file_name="health_chat_report.pdf",
        mime="application/pdf"
    )

# -------------------- FOOTER -------------------- #

st.markdown("---")
st.caption(
    "⚠ For educational purposes only. Not a substitute for professional medical advice."
)
