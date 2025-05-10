import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pdfplumber

# --- Configuration ---
MODEL_NAME = "gemini-2.0-flash"# Or "gemini-pro"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your .env file or environment variables.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🚨 Error configuring Google AI: {e}")
    st.stop()



# --- Helper Function to Extract Text from PDF ---
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: # Add text only if extract_text() returned something
                        full_text += page_text + "\n\n" # Add newline between pages
                return full_text.strip() if full_text else "No text could be extracted from this PDF."
        except Exception as e:
            return f"Error reading PDF: {e}. Is it a valid PDF file?"
    return None

# --- Initialize chat model and session ---
if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"🚨 Error initializing GenerativeModel ({MODEL_NAME}): {e}")
        st.stop()

if "chat_session" not in st.session_state:
    if "model" in st.session_state:
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])
    else:
        st.error("Model not initialized, cannot start chat.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text_context" not in st.session_state:
    st.session_state.pdf_text_context = None # To store extracted PDF text

# --- App UI ---
st.set_page_config(page_title="Gemini Chatbot with PDF", layout="wide")
st.title(f"🤖 Gemini Chatbot ({MODEL_NAME}) with PDF Analysis")
st.markdown(f"Upload a bank statement (PDF) and then ask questions about it!")

# --- PDF Upload and Processing ---
st.sidebar.header("📄 PDF Bank Statement Processor")
uploaded_pdf = st.sidebar.file_uploader("Upload your bank statement PDF", type="pdf")

if uploaded_pdf:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_pdf)
            if "Error reading PDF" in extracted_text or "No text could be extracted" in extracted_text:
                st.sidebar.error(extracted_text)
                st.session_state.pdf_text_context = None
            else:
                st.session_state.pdf_text_context = extracted_text
                st.sidebar.success("✅ PDF processed! You can now ask questions about its content.")
                # Optionally, display a snippet or word count
                st.sidebar.expander("Extracted Text Snippet (first 500 chars)").text(extracted_text[:500] + "...")
                st.sidebar.info(f"Total characters extracted: {len(extracted_text)}")

                # Clear previous messages if a new PDF is processed to start fresh context
                st.session_state.messages = []
                st.session_state.chat_session = st.session_state.model.start_chat(history=[])
                st.rerun() # Rerun to clear main chat and reflect PDF status


if st.session_state.pdf_text_context:
    st.info("ℹ️ PDF context is loaded. Your questions will be answered based on the uploaded document.")
    if st.sidebar.button("Clear PDF Context & Chat"):
        st.session_state.pdf_text_context = None
        st.session_state.messages = []
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])
        st.sidebar.info("PDF context and chat cleared.")
        st.rerun()


# --- Display existing messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Your message:", key="chat_input_main"):
    if "chat_session" not in st.session_state:
        st.error("Chat session not available. Please refresh.")
        st.stop()

    # Add user message to Streamlit display history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare prompt for Gemini, including PDF context if available
    final_prompt = prompt
    if st.session_state.pdf_text_context:
        # You might want to be more sophisticated here, e.g., only add context if relevant
        # or use a specific instruction.
        final_prompt = (
            f"Based on the following bank statement text, please answer the user's question.\n\n"
            f"--- BANK STATEMENT TEXT START ---\n"
            f"{st.session_state.pdf_text_context}\n"
            f"--- BANK STATEMENT TEXT END ---\n\n"
            f"User's Question: {prompt}"
        )
        # For very long PDFs, this might exceed context limits.
        # Consider chunking or summarizing for production.
        if len(final_prompt) > 28000: # Rough estimate, Gemini 1.0 Pro has ~32k token limit
             st.warning("⚠️ The PDF content combined with your question is very long. "
                        "The answer might be truncated or an error might occur. "
                        "Consider asking about specific parts or using a smaller PDF.")


    # Get response from Gemini
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Gemini is thinking... 🤔"):
                # Send the potentially augmented prompt
                response = st.session_state.chat_session.send_message(final_prompt)
                bot_reply = response.text
                message_placeholder.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    except Exception as e:
        error_message = f"❌ Error communicating with Gemini: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # No explicit st.rerun() needed here for chat input.

# --- General Reset chat (without clearing PDF context by default unless button is pressed) ---
if st.sidebar.button("🧹 Clear Chat Only (Keep PDF)"):
    if "model" in st.session_state:
        # Reset the API chat session but keep st.session_state.pdf_text_context
        st.session_state.chat_session = st.session_state.model.start_chat(history=[])
        # Clear the display messages
        st.session_state.messages = []
        st.success("Chat messages cleared! PDF context remains.")
        st.rerun()
    else:
        st.warning("Model not initialized, cannot clear chat. Please refresh.")