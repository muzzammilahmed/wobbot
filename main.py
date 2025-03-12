from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader
import os
import datetime

# Conversation imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFDirectoryLoader

# replace following line with location of your .docx file

def get_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    print(number_of_pages)

    pdf_text = ""

    for i in range(number_of_pages):

        page = reader.pages[i]
        pdf_text += page.extract_text()

    print(len(pdf_text))
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap  = 100,
    length_function = len,
    )
    data = text_splitter.split_text(pdf_text)
    return data

def create_db(docs,embedding):
    vectorStore = FAISS.from_texts(docs, embedding=embedding)
    return vectorStore,embedding

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="chatgpt-4o-latest",
        temperature=0.3
    )

#         Discarded Prompts

#        -if there is no context available that means there is no such thing  
#        -Always answer refering to source of documents provided 
#        -You can ask follow-up questions to better understand the user's inquiry in case question is incomplete and assist them in formulating a proper question before providing an answer

    prompt = ChatPromptTemplate.from_messages([
        ("system", """you are Wob Bot, the official chatbot of Wob Ag.
         strictly use the following pieces of context to answer the question at the end. Answer questions in bullet points if needed
        You always follow these guidelines:
]       -If a user asks an incomplete question or improper question, help them write a proper question by providing suggestions on what the question could be related to the input he/she provided
        -If the answer isn't available within the context, don't make up answer
        -If a user asks an irrelevant question that is not within the scope of Wob Ag, politely redirect them to inquire about Wob-related topics only. Alternatively, you can ask follow-up questions to clarify the user's query and maintain context.
        -If a user asks about the location or address of Wob, politely ask for their location first. Then, provide the addresses of all company locations in that city. If there is no office location there, simply inform them that Wob does not have an office in that area yet
        -Do not introduce examples outside of the context
        -Keep the answer concise, while shwoing all relevant information
        -if you dont know the asnwer simply say I'm not sure about it'
        -keep answers to the point
        -Please answer in the same language as the question
        -Wob only has a location in Viernheim
        -You can find Wob's career page here: https://wob.jobs.personio.de/. When asked for open opportunities, only use the oens listed here
        -Give links to specific pages and not the base URL everytime when you share a linked page: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Replace retriever with history aware retriever
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(

        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response["answer"]

def read_text_files(directory):
    text_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()
            text_list.append(text)
    return text_list


if __name__ == "__main__":

    embedding = OpenAIEmbeddings()

    # Enable the following when reading and creating the vector database for the first time
    
    # docs = read_text_files("./text_data_scrapped/")
    # vectorStore,embedding = create_db(docs,embedding)
    # vectorStore.save_local("./")
    
    loaded_vectorStore = FAISS.load_local("./", embedding,  allow_dangerous_deserialization=True)

   # loaded_vectorStore = FAISS.load_local("", embedding, allow_dangerous_deserialization=True)
    chain = create_chain(loaded_vectorStore)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="Wob Bot",page_icon = "👨‍⚖️")
    st.title("Wob Bot")

    # Get the current hour
    current_hour = datetime.datetime.now().hour

    if 'messages' not in st.session_state.keys():
        if 5 <= current_hour < 12:
            st.session_state.messages = [{'role':'assistant','content':'Good Morning, how can I help you?'}]
        elif 12 <= current_hour < 18:
            st.session_state.messages = [{'role':'assistant','content':'Good Afternoon, how can I help you?'}]
        else:
            st.session_state.messages = [{'role':'assistant','content':'Good Evening, how can I help you?'}]


    for messages in st.session_state.messages:
        with st.chat_message(messages['role']):
            st.write(messages['content'])

    user_prompt = st.chat_input()
    if user_prompt is not None:
        st.session_state.messages.append({'role':'user','content':user_prompt})
        with st.chat_message('user'):
            st.write(user_prompt)
    #print(st.session_state.messages[-1]['role'])

    if user_prompt is not None:
        with st.chat_message('assistant'):
            with st.spinner('Loading...'):
                response = process_chat(chain, user_prompt, st.session_state.chat_history)
                st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.write(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
