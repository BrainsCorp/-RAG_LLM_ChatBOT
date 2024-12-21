import streamlit as st
from model.llm import open_ai_model
from utils.retriever import LangChainRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from utils.reasoning_query import Reasoning_engine
from streamlit_pdf_viewer import pdf_viewer

import uuid
import nltk
from nltk.data import find
from nltk.corpus import stopwords


# Initialize
prompt = hub.pull("rlm/rag-prompt")
retriever = LangChainRetriever(vector_db_path='vector_db')

# Initialize sources list
if "sources" not in st.session_state:
    st.session_state.sources = []

@st.cache_resource
def download_stopwords():
    try:
        find('corpus/stopwords')
    except LookupError:
        nltk.download('stopwords')

# --- Callbacks ---
def format_docs(docs):
    """Formats document content for display."""
    return "\n\n".join([
        f"{doc.metadata.get('Course code')}\n{doc.metadata.get('course name')}\n{doc.page_content}\n{doc.metadata.get('descri')}\n{doc.metadata.get('outcomes_x')}"
        for doc in docs
    ])

def highlight_relevant_parts(text, question):
    """Highlights relevant parts of the text based on the question."""
    highlighted_text = text
    download_stopwords()
    stop_words = set(stopwords.words('english'))
    words = set(question.split()).difference(stop_words)

    for word in words:
        highlighted_text = highlighted_text.replace(
            word,
            f'<span style="text-decoration: underline; color: lightblue;">{word}</span>',
            1
        )
    return highlighted_text

def add_file():
    """Add uploaded files to the sources."""
    uploaded_files = st.session_state.get("uploaded_files", [])
    for uploaded_file in uploaded_files:
        if uploaded_file and uploaded_file.name not in [source["name"] for source in st.session_state.sources]:
            st.session_state.sources.append({"type": "file", "name": uploaded_file.name, "data": uploaded_file})

def add_hyperlink():
    """Add a hyperlink to the sources."""
    hyperlink = st.session_state.get("hyperlink", "")
    if hyperlink and hyperlink not in [source["data"] for source in st.session_state.sources if source["type"] == "link"]:
        st.session_state.sources.append({"type": "link", "data": hyperlink})

def remove_source(index):
    """Remove a source by index."""
    del st.session_state.sources[index]

# --- UI ---

# Streamlit App title
st.title("Course Wizard 🧙‍♂️")

# Sidebar
with st.sidebar:
    st.subheader("📚 Add your course materials")
    uploaded_files = st.file_uploader("Choose files", type=['pdf', 'doc', 'docx'], accept_multiple_files=True, on_change=add_file, key="uploaded_files")
    st.text_input("Paste a hyperlink", on_change=add_hyperlink, key="hyperlink")

    st.subheader("Your sources:")
    for i, source in enumerate(st.session_state.sources):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.write(f"📄 {source['name']}" if source["type"] == "file" else f"🔗 {source['data']}")
        with col2:
            st.button("❌", key=f"remove_{i}", on_click=remove_source, args=(i,))
    
    st.subheader("📄 Uploaded PDF Viewer")
    for source in st.session_state.sources:
        if source["type"] == "file" and source["name"].endswith(".pdf"):
            with st.expander(f"View: {source['name']}"):
                # Display PDF using Streamlit's built-in viewer
                pdf_data = source["data"].read() # Read the file as bytes
                pdf_viewer(pdf_data, width=200)


# User input
user_input = st.text_area("I'm your assistant to search anything in Humber's DB!")

if st.button("Generate"):
    if user_input:
        try:
            # Reasoning and retrieval
            question = user_input
            filters = Reasoning_engine(question).answer

            if filters['source']:
                documents, scores = zip(*retriever.query_retriever(question, filters))
                context = format_docs(documents) if filters['source'] == 'merged_humber_courses.csv' else "\n\n".join(doc.page_content for doc in documents)
            else:
                context = "Answer from your own knowledge"

            # Response generation
            template = f"""You are an expert assistant for Humber College faculty. 
            Use the provided context to answer questions accurately. Question: {question} Context: {context} Answer:"""
            response = open_ai_model(template)
            st.markdown(rf"{response.content}")

            # Display relevant documents in a responsive grid
            if filters['source'] == 'merged_humber_courses.csv':
                st.subheader("Relevant Documents:")
                for i in range(0, len(documents), 3):
                    cols = st.columns(min(3, len(documents) - i))
                    for j, col in enumerate(cols):
                        doc = documents[i + j]
                        with col:
                            st.markdown(highlight_relevant_parts(doc.page_content, question), unsafe_allow_html=True)
                            st.write(f"**Course Code:** {doc.metadata.get('Course code')}\n{doc.metadata.get('descri')[:100]}\n")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Write your prompt!")
