import streamlit as st
import os
import io
import pandas as pd
import ast
from PyPDF2 import PdfReader
import docx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from multiprocessing import Pool
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
import logging
from typing import Union
from datetime import datetime
import spacy  # Added for Named Entity Recognition (NER)
from textract import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from config import GOOGLE_API_KEY

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
nlp = spacy.load("en_core_web_sm") 
def extract_text(file): # Added for various file types text extraction
    text = ""
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text = extract_text_from_pdf(file)
    elif file_extension in ["doc", "docx"]:
        text = extract_text_from_docx(file)
    return text

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        handle_file_processing_error("PDF", e)
    return text

# Function to extract text from a Word file
def extract_text_from_docx(docx_file):
    text = ""
    try:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        handle_file_processing_error("DOCX", e)
    return text

# Function to handle file processing errors
def handle_file_processing_error(file_type: str, error: Exception):
    st.error(f"Error processing {file_type} file: {error}")
    logger.exception(f"Error processing {file_type} file", exc_info=True)

# Function to handle AI model interaction errors
def handle_model_interaction_error(error: Exception):
    st.error(f"Error interacting with AI model: {error}")
    logger.exception("Error interacting with AI model", exc_info=True)

# Create a logger
logger = logging.getLogger(__name__)

# Function to handle input validation errors
def validate_user_input(user_input: Union[st.file_uploader, str]):
    if not user_input:
        st.warning("Please provide valid input.")
        return False
    return True

# Function to handle user feedback
def handle_user_feedback(feedback: str):
    st.success("Thank you for your feedback!")

# Function to log important events and interactions
def log_event(event: str):
    logger.info(event)

# Function to add a pre-prompt to the MARS
def add_pre_prompt(prompt, context, data_types, **kwargs):
    now = datetime.now()
    pre_prompt = f"""
GENERAL INSTRUCTIONS: *NEVER INCLUDE THE USER'S QUESTION OR MESSAGE* PROVIDE A DETAILED RESPONSE OF AT LEAST 50 WORDS, 
ENSURING CLARITY AND CONCISENESS. CITE SOURCES OR PROVIDE REFERENCES WHENEVER APPLICABLE. 
MAKE SURE YOU DO NOT PROVIDE USELESS INFORMATION ABOUT THE TOPIC.
(Today is {now.strftime("%d/%m/%Y %H:%M:%S")}, your name is KARM: Knowledge-Enhanced Assistance for Resume Management" 
by Sahil Bhoite KARM: Knowledge-Enhanced Assistance for Resume Management" is an AI assistant focused on helping users with tasks related to resume management such as resume parsing, extraction of key information, skills analysis etc. I was created by Anthrop ic to be helpful, harmless, and honest in all interactions. How may I assist you today?)
PREVIOUS MESSAGE: ({context})
USER ASKED: {prompt}
"""

    for data_type in data_types:
        if data_type == "data":
            solution = kwargs.get('solution')
            if len(solution.split()) < 50:
                pre_prompt += f"""
ENHANCE THE ANSWER BY:
- Providing concrete examples or real-world scenarios to illustrate concepts effectively.
- Structuring the response logically with bullet points or step-by-step explanations for clarity.
- Avoiding generic statements and ensuring depth in analysis.
- Understanding the user's context and tailoring the response accordingly.
- Incorporating visuals or diagrams where applicable to enhance comprehension.
"""

        elif data_type == "context":
            solution = kwargs.get('solution')
            if len(solution.split()) < 15:
                pre_prompt += f"HERE IS THE CONTEXT YOU ASKED FOR: {solution}\n"
                pre_prompt += f"EXPAND ON THE CONTEXT TO PROVIDE MORE DETAILS OR RELATED INFORMATION."
            pre_prompt += f"""
EXPAND ON THE ANSWER BY:
- Providing additional context or background information to enrich understanding.
- Utilizing analogies or real-world examples to simplify complex concepts.
- Clarifying common misconceptions or areas of confusion preemptively.
- Encouraging critical thinking by posing thought-provoking questions.
- Recommending supplementary resources or further reading for continued learning.
"""
        elif data_type == "pdf":
            pdf_content = kwargs.get('pdf_content')
            if len(solution.split()) < 15:
                pre_prompt += f"HERE IS THE CONTENT OF THE PDF: {solution}\n"
                pre_prompt += f"PROVIDE FURTHER ANALYSIS OR CONTEXTUAL INFORMATION FOR A MORE IN-DEPTH DISCUSSION."
            pre_prompt += f"""
ENHANCE THE ANSWER BY:
- The uploaded files are either resume/CV or job descriptions. your JOB is to Assist the user in extracting key information from these files and analyzing them. You can do this by identifying:
- Analyze the  data from sites available and use it to enhance the talent engagement and hiring process and provide recommendations to the user on how to improve their resume/CV or job description based on industry standards and best practices.
- This could involve analyzing job descriptions, company information, and candidate profiles to match candidates with jobs that are not only relevant to their skills and experience but also aligned with their career goals and preferences. 
- consider the dynamic nature of the job market and the evolving needs of the companies and the candidates.
- Extract key information from resumes/CVs like contact details, work experience, skills, education etc and analyze them to identify:
  - Relevant skills and experiences based on the job role
"""
        elif data_type == "other":
            file_info = kwargs.get('file_info')
            if len(solution.split()) < 15:
                pre_prompt += f"HERE IS THE CONTENT FROM THE FILE: {solution}\n"
                pre_prompt += f"EXPAND ON THE PROVIDED INFORMATION TO OFFER MORE INSIGHTS OR CONTEXT."
            pre_prompt += f"""
EXPAND THE ANSWER BY:
- Integrating additional details or perspectives from the file content.
- Contextualizing the extracted information within a broader thematic framework.
- Engaging with the user's inquiry through thoughtful analysis and commentary.
- Stimulating curiosity and critical thinking with probing questions or hypotheses.
- Proposing potential applications or implications for practical relevance.
"""
        elif data_type == "resume":
            resume_content = kwargs.get('resume_content')
            pre_prompt += f"""
-Hey KARMA Act Like a skilled or very experience ATS(Application Tracking System) and evaluate the resume content shared by the user and provide insightful feedback on:
- if there is mention of Github, LinkedIn or Contact details then its resume and judge it on following paramenters 
- Here are 10 parameters that can be used to judge resumes for tech jobs, each with a weight of 10 points out of a total of 100 points do that when resume is deted:
- Technical Skills (10 points): Evaluate the breadth and depth of the applicant's technical skills, including programming languages, frameworks, tools, and technologies relevant to the specific tech role or industry.
- Projects and Contributions (10 points): Assess the presence and quality of any notable projects, contributions to open-source projects, personal projects, or coding samples showcased in the resume, demonstrating practical experience and problem-solving abilities.
- Education and Certifications (10 points): Evaluate the relevance and quality of the applicant's educational background, including degrees, majors, relevant coursework, and any industry-recognized certifications or training in technical domains.
- Problem-Solving and Analytical Skills (10 points): Assess the applicant's ability to demonstrate problem-solving and analytical skills through the description of their work experiences, projects, or accomplishments, highlighting their ability to tackle complex technical challenges.
- Version Control and Collaboration (10 projects): Evaluate the applicant's familiarity and experience with version control systems (e.g., Git) and collaboration tools (e.g., GitHub, Bitbucket), which are essential for software development and teamwork in tech environments.
- Technical Writing and Documentation (10 points): Assess the applicant's ability to communicate technical concepts effectively through clear and concise writing, as demonstrated in their resume, cover letter, or any technical writing samples provided.
- Continuous Learning and Professional Development (10 points): Evaluate the applicant's commitment to continuous learning and professional development, as evidenced by their participation in relevant workshops, conferences, online courses, or personal projects aimed at expanding their technical knowledge.
- Quantifiable Achievements (10 points): Assess the presence and quality of quantifiable achievements or measurable results mentioned in the resume, demonstrating the applicant's impact and value in technical roles or projects.
- Attention to Detail (10 points): Evaluate the applicant's attention to detail, as demonstrated by the accuracy, consistency, and professionalism of their resume formatting, language use, and overall presentation.
- Industry and Domain Knowledge (10 points): Assess the applicant's understanding and familiarity with the specific industry or domain related to the tech role, including relevant buzzwords, trends, and best practices, showcasing their ability to align their technical skills with the business needs.
"""           
    return pre_prompt

# Main Function for KARM: Knowledge-Enhanced Assistance for Resume Management" 
def main():
    st.set_page_config(page_title="K.A.R.M 📈", layout="wide")
    st.header("Knowledge-Enhanced Assistance for Resume Management ")
    user_question = st.chat_input("Ask Questions about Job Description and Your Resume")

    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.files_uploaded = False

    with st.sidebar:
        st.title("K.A.R.M 📈")
        st.subheader("Upload your Resume or CV Files here")
        files = st.file_uploader("Upload your Files and Click on the NEXT Button", accept_multiple_files=True)

        if st.button("NEXT"):
            if validate_user_input(files):
                with st.spinner("Processing your Resume Files..."):
                    raw_text = ""
                    for file in files:
                        try:
                            text = extract_text(file)
                            raw_text += text
                        except Exception as e:
                            handle_file_processing_error(file.name.split(".")[-1].lower(), e)

                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.session_state.files_uploaded = True
                    if st.session_state.conversation is None:
                        st.session_state.files_uploaded = False
                        return
                    st.success("Processing Done!")
            else:
                st.warning("Please upload at least one file.")

    if user_question:
        user_input(user_question)

    if not st.session_state.files_uploaded:
        st.warning("Upload your resume or CV files and Paste the link with Job Description to get started")
    elif st.session_state.files_uploaded and not files:
        st.session_state.files_uploaded = False


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
    
# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create a conversational chain
def get_conversational_chain(vector_store):
    try:
        llm = GooglePalm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        return conversation_chain
    except NotImplementedError as e:
        st.error(" Please try again tap on 'NEXT'")
        return None
    except Exception as e:
        handle_model_interaction_error(e)
        st.error(" Please try again tap on 'NEXT'")
        return None


def user_input(user_question):
    if st.session_state.conversation:
        try:
            # Check if the user's message contains a feedback mention
            is_feedback = "@karm" in user_question.lower()
            is_feedback = is_feedback or "@karm" in user_question.upper()
            
            # If the message contains a feedback mention, handle it
            if is_feedback:
                handle_user_feedback(user_question)
            else:
                # Otherwise, proceed with the conversation
                response = st.session_state.conversation({'question': user_question})
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        if isinstance(message, HumanMessage):
                            with st.chat_message("User"):
                                st.write(message.content)
                        elif isinstance(message, AIMessage):
                            with st.chat_message("AI"):
                                st.write(message.content)
                    
                    # Auto-scroll to the end of the chat with smooth behavior
                    st.markdown(
                        """
                        <script>
                        var element = document.getElementById("end-of-chat");
                        element.scrollIntoView({behavior: "smooth"});
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            handle_model_interaction_error(e)
            st.error("An error occurred during conversation. Please try again.")
    else:
        st.warning("Please upload files and click 'NEXT' to start the conversation.")


# Running the main function
if __name__ == "__main__":
    main()
