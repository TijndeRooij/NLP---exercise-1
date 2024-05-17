import pandas as pd
import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Initialize the model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load vectors from disk
vectors = np.load('Vectors/vectors.npy')

# Load FAISS index
index = faiss.read_index('Vectors/faiss_index.index')

# Function to retrieve movie info based on a query
def get_vectors(query):
    query_vector = embedder.encode([query])
    D, I = index.search(query_vector, 4)
    return [df.iloc[I[0][1]], df.iloc[I[0][2]], df.iloc[I[0][3]]]

# Get info gets all the information about the movie in the used dataset.
def get_info(movie_name):
    return df[df['title'] == movie_name]

# Transform to prompt adds all the known info from the info variable to a string which will be used in the prompt for the NLP-model
def transform_to_prompt(info):
    return f'The title of the movie is: {info['title'].iloc[0]}, This {info['type'].iloc[0]} is about {info['description'].iloc[0]} The whole {info['type'].iloc[0]} takes {info['duration'].iloc[0]}. The genre of the {info['type'].iloc[0]} is listed as {info['listed_in'].iloc[0]}. The {info['type'].iloc[0]} was released in {info['release_year'].iloc[0]}. The cast of the {info['type'].iloc[0]} is {info['cast'].iloc[0]} and the director of the {info['type'].iloc[0]} is {info['director'].iloc[0]}.'


# Streamlit application having a NLP-model

# Imports the model
model = Ollama(model="llama3")

# Imports the dataset
df = pd.read_csv(r'Data\Data.csv')

# Sets the title list
title_list = df['title'].tolist()

# Sets the title of the page
st.title('TV expert')

# Uses the title list to make a selectbox in which all the movies can be selected
title = st.selectbox('🍿Pick a movie or TV show you would like know more about.🍿', 
                     title_list, 
                     index=None, 
                     label_visibility='collapsed', 
                     placeholder='Select your movie or TV show here!')

# Title should be None when nothing has been selcted yet
if "title" not in st.session_state:
    st.session_state.title = None

# Initialize conversation memory
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(
        llm=model,
        memory=ConversationBufferMemory()
    )

# There should be no messages when none have been asked
if "messages" not in st.session_state:
    st.session_state.messages = []

# When something has been selected
if title and title != st.session_state.title:
    # Set the selected title
    st.session_state.title = title

    # Get and store movie info
    st.session_state.info = get_info(title)
    movie_info = st.session_state.info
    movie_simularities = get_vectors(movie_info['description'].iloc[0])

    if not movie_info.empty:
        # Display initial message
        st.session_state.conversation_chain.memory.clear()  # Clear previous memory

        initial_message = f'Please ask me questions about the {movie_info["type"].iloc[0]} {movie_info["title"].iloc[0]}. If you want to ask me questions about simulair titles, feel free to select one of these: \n  - **{movie_simularities[0]['title']}**, \n  - **{movie_simularities[1]['title']}**, \n  - **{movie_simularities[2]['title']}**.'
        st.session_state.conversation_chain.memory.chat_memory.add_message(AIMessage(content=initial_message))
        st.session_state.messages.append({"role": "assistant",
                                          "content": initial_message})
    else:
        st.title('Movie not known')

# Make sure the messages are clearly shown after they have been asked
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# When a question is asked
if question := st.chat_input('Type here!'):
    # Sets up question
    with st.chat_message('user'):
        st.markdown(question)
    st.session_state.conversation_chain.memory.chat_memory.add_message(HumanMessage(question))
    st.session_state.messages.append({"role": "user",
                                      "content": question})

    try:
        # Initialises variable for the prompt
        listed_type = st.session_state.info['type'].iloc[0]
        listed_title = st.session_state.info['title'].iloc[0]
        info_string = transform_to_prompt(st.session_state.info)

        # Create the prompt
        prompt_template: str = """/
        You are a {listed_type} expert which only know about the {listed_type}: {listed_title}. You are going to answer every question of the user as an expert about this {listed_type}. You are going to use this discription: {info}. With this information answer this question: {question}.
        """
        prompt = PromptTemplate.from_template(template=prompt_template)

        # Format the prompt to add variable values
        prompt_formatted_str: str = prompt.format(question=question,    
                                                  listed_type=listed_type,
                                                  listed_title=listed_title,
                                                  info=info_string)

        # Get response from the conversation chain
        response = st.session_state.conversation_chain.run(input=prompt_formatted_str)

        # Ensure response is a string
        if isinstance(response, list):
            response = "".join(response)

        # Sets up response
        with st.chat_message('assistant'):
            st.markdown(response)
        st.session_state.conversation_chain.memory.chat_memory.add_message(AIMessage(response))
        st.session_state.messages.append({"role": "assistant",
                                          "content": response})
    except:
        # Error
        st.title('Please pick a movie first.')  