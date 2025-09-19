import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from together import Together
from instagrapi import Client
import tempfile

# Supported models
MODEL_CHOICES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile"
]

# Sidebar: API key and model
api_key_together = st.sidebar.text_input("TOGETHER_API_KEY", type="password", key="api_key_together")
model_image = st.sidebar.text_input("Image Model", value="black-forest-labs/FLUX.1-schnell-Free", key="model_image")

st.sidebar.divider()
api_key = st.sidebar.text_input("Groq API Key", type="password", key="api_key")
model = st.sidebar.selectbox("Model", MODEL_CHOICES, index=1, key="model")

st.sidebar.divider()
ig_user = st.sidebar.text_input("Instagram User ID", key="ig_user")
ig_password = st.sidebar.text_input("Instagram Password", type="password", key="ig_password")

if "solution" not in st.session_state:
    st.session_state.solution = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

st.title("Simple Insta")

def get_solution(prompt, api_key, model, api_key_together, model_image):

    os.environ["GROQ_API_KEY"] = api_key
    os.environ["TOGETHER_API_KEY"] = api_key_together
    # Generate image with Together API
    client_together = Together(api_key=api_key_together)
    response = client_together.images.generate(
        prompt=prompt,
        model=model_image,
        steps= 1 # you can adjust steps as needed
    )
    img_url = response.data[0].url
    st.image(img_url, caption=f"Prompt: {prompt}")

    # Generate caption/hashtags with ChatGroq
    llm = ChatGroq(model=model)
    template = ChatPromptTemplate.from_messages([
        ("system",  f"Write a stylish Instagram hashtag and an engaging caption within 100 words only for this image prompt: '{prompt}'"),
        ("human", "{input}"),
    ])
    chain = template | llm
    result = chain.invoke({"input": prompt})
    return result.content, img_url

# Only show prompt input if API key is entered
if api_key and api_key_together and ig_user and ig_password:
    prompt = st.text_input("Describe your image:", value=st.session_state.last_prompt)
    if st.button("OK"):
        st.session_state.solution, st.session_state.img_url = get_solution(prompt, api_key, model, api_key_together, model_image)
        st.session_state.last_prompt = prompt

    if st.session_state.solution and st.session_state.img_url:
        st.markdown("**Caption:**")
        st.write(st.session_state.solution)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Recreate"):
                st.session_state.solution, st.session_state.img_url = get_solution(
                    st.session_state.last_prompt,
                    api_key,
                    model,
                    api_key_together,
                    model_image
                )
                st.rerun()
        with col2:
            if st.button("Post to Instagram"):
                cl = Client()
                try:
                    cl.login(ig_user, ig_password)
                    img_data = requests.get(st.session_state.img_url).content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(img_data)
                        tmp_file_path = tmp_file.name
                    cl.photo_upload(tmp_file_path, st.session_state.solution)
                    st.success("Posted to Instagram successfully!")
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.error(f"Error posting to Instagram: {e}")
else:
    st.info("Please enter your Groq and Together API key and your Instagram User ID and Password in the sidebar to begin.")

