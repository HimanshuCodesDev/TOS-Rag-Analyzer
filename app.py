import streamlit as st
import os
import re
import json
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from groq import RateLimitError
from langchain.embeddings import SentenceTransformerEmbeddings
import plotly.graph_objects as go

load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Terms of Service Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("Terms of Service Analyzer")

# Debug information
st.sidebar.title("Debug Information")
debug_info = st.sidebar.empty()

def update_debug_info(message):
    debug_info.text(message)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def vector_embedding(documents):
    update_debug_info("Creating vector embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    if not final_documents:
        st.error("No documents to process.")
        return None

    document_texts = [doc.page_content for doc in final_documents]

    vectors = FAISS.from_texts(document_texts, embeddings)
    update_debug_info("Vector embeddings created successfully.")
    return vectors

def load_document(uploaded_file):
    update_debug_info(f"Loading document: {uploaded_file.name}")
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        update_debug_info(f"PDF loaded. Text length: {len(text)} characters")
        return text
    elif uploaded_file.name.endswith('.txt'):
        text = uploaded_file.read().decode('utf-8')
        update_debug_info(f"Text file loaded. Text length: {len(text)} characters")
        return text
    else:
        st.error("Unsupported file format. Please upload a .txt or .pdf file.")
        return None

MAX_TOKENS_PER_MINUTE = 30000
last_request_time = time.time()
tokens_used_in_window = 0

def rate_limited_invoke(chain, context, input_text):
    global last_request_time
    global tokens_used_in_window

    update_debug_info("Invoking LLM chain...")
    tokens_to_use = len(context) + len(input_text)

    if time.time() - last_request_time >= 60:
        tokens_used_in_window = 0
        last_request_time = time.time()

    if tokens_used_in_window + tokens_to_use > MAX_TOKENS_PER_MINUTE:
        sleep_time = 60 - (time.time() - last_request_time)
        update_debug_info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        tokens_used_in_window = 0
        last_request_time = time.time()

    try:
        response = chain.invoke({'context': context, 'input': input_text})
        tokens_used_in_window += tokens_to_use
        update_debug_info("LLM chain invoked successfully.")
        return response
    except RateLimitError as e:
        st.error(f"Rate limit exceeded: {e}")
        return None
    except Exception as e:
        st.error(f"Error invoking LLM chain: {e}")
        return None

def process_clauses(llm_response):
    update_debug_info("Processing clauses from LLM response...")
    clauses = []
    clause_pattern = re.compile(r'\*\*Clause \d+: (.+?)\*\*\n\nFairness: (.+?)\nExplanation: (.+?)(?=\n\n\*\*Clause|\Z)', re.DOTALL)
    
    matches = clause_pattern.finditer(llm_response)
    for match in matches:
        clause = {
            "name": match.group(1).strip(),
            "fairness": match.group(2).strip(),
            "details": match.group(3).strip()
        }
        clauses.append(clause)
    
    update_debug_info(f"Processed {len(clauses)} clauses.")
    return clauses

def extract_summary_and_compliance(llm_response):
    summary_match = re.search(r'\*\*Overall Summary:\*\*(.*?)(?=\*\*Legal Compliance:|$)', llm_response, re.DOTALL)
    compliance_match = re.search(r'\*\*Legal Compliance:\*\*(.*)', llm_response, re.DOTALL)
    
    summary = summary_match.group(1).strip() if summary_match else ""
    compliance = compliance_match.group(1).strip() if compliance_match else ""
    
    return summary, compliance

def main():

    uploaded_file = st.file_uploader("Upload a Terms of Service document", type=["pdf", "txt"])

    if uploaded_file is not None:
        document_text = load_document(uploaded_file)
        if document_text is None:
            st.error("Failed to load the document. Please try again.")
            return

        st.info("Document loaded successfully. Analyzing...")

        documents = [Document(page_content=document_text)]
        vector_store = vector_embedding(documents)
        
        if vector_store is None:
            st.error("Failed to create vector embeddings. Please try again.")
            return

        retriever = vector_store.as_retriever()
        
        prompt = ChatPromptTemplate.from_template("""
        You are a legal expert specializing in analyzing Terms of Service agreements. Analyze the following Terms of Service and provide a detailed breakdown of the clauses, their fairness, and compliance with laws. Use the following format for each clause:

        **Clause [number]: [Name of the clause]**

        Fairness: [Clearly Fair | Potentially Unfair | Clearly Unfair]
        Explanation: [Brief explanation for the classification, including any legal implications]

        After analyzing all clauses, provide:

        **Overall Summary:**
        [A brief summary of the overall fairness of the Terms of Service]

        **Legal Compliance:**
        [Highlight any non-compliance issues with specific laws]

        Here's the Terms of Service to analyze:

        {context}

        Human: Analyze the above Terms of Service.

        Assistant: Here's my analysis of the Terms of Service:
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = rate_limited_invoke(retrieval_chain, document_text, "Analyze the Terms of Service")
        
        if response is None or 'answer' not in response:
            st.error("Failed to analyze the document. Please try again.")
            return

        llm_response = response['answer']
        update_debug_info(f"LLM Response received. Length: {len(llm_response)} characters")
        
        clauses = process_clauses(llm_response)
        summary, compliance = extract_summary_and_compliance(llm_response)
        
        clearly_fair = sum(1 for clause in clauses if clause['fairness'] == "Clearly Fair")
        potentially_unfair = sum(1 for clause in clauses if clause['fairness'] == "Potentially Unfair")
        clearly_unfair = sum(1 for clause in clauses if clause['fairness'] == "Clearly Unfair")
        total_clauses = len(clauses)

        st.header("Analysis Results")

        # Display the summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Clearly Fair", value=clearly_fair)
        col2.metric(label="Potentially Unfair", value=potentially_unfair)
        col3.metric(label="Clearly Unfair", value=clearly_unfair)

        # Display overall result
        st.subheader("Total Clauses Analyzed")
        st.info(f"{total_clauses}")

        if total_clauses > 0:
            # Display pie chart
            labels = ['Clearly Fair', 'Potentially Unfair', 'Clearly Unfair']
            values = [clearly_fair, potentially_unfair, clearly_unfair]
            colors = ['#4CAF50', '#FFC107', '#F44336']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])
            fig.update_layout(title_text="Distribution of Clause Fairness")
            st.plotly_chart(fig)

            # Display detailed clause breakdown
            st.subheader("Detailed Clause Analysis")
            
            for clause in clauses:
                with st.expander(f"{clause['name']} - {clause['fairness']}"):
                    if clause['fairness'] == "Clearly Fair":
                        color = "#4CAF50"
                    elif clause['fairness'] == "Potentially Unfair":
                        color = "#FFC107"
                    else:
                        color = "#F44336"
                    
                    st.markdown(f"<p style='color:{color};'><strong>Fairness:</strong> {clause['fairness']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Explanation:</strong> {clause['details']}</p>", unsafe_allow_html=True)

            # Display summary and compliance
            st.subheader("Overall Summary")
            st.write(summary)

            st.subheader("Legal Compliance")
            st.write(compliance)
        else:
            st.warning("No clauses were identified in the analysis. This could be due to an error in processing or an unexpected response format from the AI model.")
            st.subheader("Raw AI Response")
            st.code(llm_response)

if __name__ == "__main__":
    main()