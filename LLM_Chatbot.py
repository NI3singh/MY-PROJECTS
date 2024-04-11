import streamlit as st
from langchain.llms import OpenAI

# Function to get the answer from OpenAI
def get_answer_from_open_ai(question: str) -> str:
    openai_api_key = "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"  # OpenAI API key
    llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai_api_key)
    return llm(question)

# Streamlit configuration
st.set_page_config(page_title="QnA app with Langchain and OpenAI")
st.header("Question And Answering App")

# User input
question = st.text_input('You: ', key='input')

# Get the response when the user clicks the button
submit = st.button('Get the answer')

if submit:
    response = get_answer_from_open_ai(question)
    st.write(response)
