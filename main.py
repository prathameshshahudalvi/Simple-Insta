import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from together import Together
from instagrapi import Client
from instagrapi.exceptions import TwoFactorRequired
import tempfile

# Global registry for 2FA handler
_2FA_HANDLER = None


# Global Instagram client
cl = None

def set_2fa_handler():
    """Set the global 2FA handler"""
    global _2FA_HANDLER
    _2FA_HANDLER = handle_2fa_code

def get_instagram_client():
    """Get or create Instagram client instance"""
    global cl
    if cl is None:
        cl = Client()
        cl.challenge_code_handler = lambda username, choice: handle_2fa_code(username, choice)
    return cl

def set_2fa_handler():
    """Set the global 2FA handler"""
    global _2FA_HANDLER
    _2FA_HANDLER = handle_2fa_code

# Supported models
MODEL_CHOICES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile"
]

IMAGE_MODEL_CHOICES = [
    "black-forest-labs/FLUX.1-kontext-pro",
    "black-forest-labs/FLUX.1-krea-dev",
    "black-forest-labs/FLUX.1-dev-lora",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-schnell-Free",
]

# Sidebar: API key and model
api_key_together = st.sidebar.text_input("TOGETHER_API_KEY", type="password")
model_image = st.sidebar.selectbox("Image Model", IMAGE_MODEL_CHOICES, index=0)

st.sidebar.divider()
api_key = st.sidebar.text_input("Groq API Key", type="password")
model = st.sidebar.selectbox("Model", MODEL_CHOICES, index=1)

st.sidebar.divider()
ig_user = st.sidebar.text_input("Instagram User ID")
ig_password = st.sidebar.text_input("Instagram Password", type="password")

# Initialize session state variables
if "solution" not in st.session_state:
    st.session_state.solution = ""
if "img_url" not in st.session_state:
    st.session_state.img_url = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "ready_to_post" not in st.session_state:
    st.session_state.ready_to_post = False
if "for_first_2fa_entry" not in st.session_state:
    st.session_state.for_first_2fa_entry = True
if "already_requested_2fa" not in st.session_state:
    st.session_state.already_requested_2fa = False
if "last_checked_2fa" not in st.session_state:
    st.session_state.last_checked_2fa = None


st.title("Simple Insta")


def handle_2fa_code(username: str, choice=None):
    """
    Handle 2FA code input through Streamlit interface
    
    Parameters
    ----------
    username: str
        Instagram username
    choice: str, optional
        Method of 2FA (sms or email)
        
    Returns
    -------
    str or None
        The 6-digit code if available, None if waiting for input
    """
    # Initialize session state for 2FA
    
    # Check if we already have a code stored
    if f"2fa_code_{username}" in st.session_state and st.session_state[f"2fa_code_{username}"]:
        code = st.session_state[f"2fa_code_{username}"]
        # Clear the code after using it
        del st.session_state[f"2fa_code_{username}"]
        if f"2fa_required_{username}" in st.session_state:
            del st.session_state[f"2fa_required_{username}"]
        return code
    
    # Mark that 2FA is required
    if f"2fa_required_{username}" not in st.session_state:
        st.session_state[f"2fa_required_{username}"] = True
        st.session_state[f"2fa_code_{username}"] = ""
    
    # Show 2FA input form
    st.warning(f"🔐 Two-factor authentication required for {username}")
    
    code = st.text_input(
        f"Enter verification code (6 digits) for {username} ({choice or 'SMS/Email'}):", 
        key=f"2fa_input_{username}",
        max_chars=6,
        help="Check your phone or email for the verification code"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Submit Code", key=f"2fa_submit_{username}"):
            if code and code.isdigit() and len(code) == 6:
                st.session_state[f"2fa_code_{username}"] = code
                st.session_state[f"2fa_required_{username}"] = False
                st.success("✅ Code submitted successfully! Please Wait...")
                st.session_state.last_checked_2fa = True
                return None  # Don't rerun, let user click the button again
            else:
                st.error("❌ Please enter a valid 6-digit code")
    
    with col2:
        if st.button(f"Cancel", key=f"2fa_cancel_{username}"):
            # Reset 2FA state
            if f"2fa_required_{username}" in st.session_state:
                del st.session_state[f"2fa_required_{username}"]
            if f"2fa_code_{username}" in st.session_state:
                del st.session_state[f"2fa_code_{username}"]
            st.session_state.for_first_2fa_entry = True
            st.info("2FA cancelled")
            return None
    
    # Return None to indicate we're waiting for user input
    st.session_state.already_requested_2fa = True
    st.stop()
    return None

# Set up the global 2FA handler
set_2fa_handler()

def get_solution(prompt, api_key, model, api_key_together, model_image):
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["TOGETHER_API_KEY"] = api_key_together

    # Generate image with Together API
    client_together = Together(api_key=api_key_together)
    response = client_together.images.generate(
        prompt=prompt,
        model=model_image,
        steps=1
    )
    img_url = response.data[0].url

    # Generate caption/hashtags with ChatGroq
    llm = ChatGroq(model=model)
    template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an Instagram content creator. Write a short, engaging caption for this image idea: '{prompt}'. Make it fun and natural for Instagram users. Include stylish hashtags at the end, but do not label them or count words. Keep the tone conversational and engaging.Include stylish hashtags at the end, but do not label them or count words.Keep the tone conversational and engaging."""),
        ("human", "{input}"),
    ])
    chain = template | llm
    result = chain.invoke({"input": prompt})
    return result.content, img_url


prompt = st.text_input("Describe your image:", value=st.session_state.last_prompt)

# Generate on button press only
if st.button("Generate"):
    try:
        st.session_state.solution, st.session_state.img_url = get_solution(
            prompt, api_key, model, api_key_together, model_image
        )
        st.session_state.last_prompt = prompt
        st.session_state.ready_to_post = True
    except Exception as e:
        st.error(f"Error generating: {e}")

# Show results if available
if st.session_state.ready_to_post:
    st.markdown("**Caption:**")
    st.write(st.session_state.solution)
    st.image(st.session_state.img_url)

    if st.session_state.already_requested_2fa:
        handle_2fa_code(ig_user)
   
    if not st.session_state.last_checked_2fa:
        if st.button("Post to Instagram") or st.session_state.already_requested_2fa:
            if not ig_user or not ig_password:
                st.error("Please enter Instagram credentials in the sidebar")
            else:
                # Check if 2FA is currently required and no code is available
                if f"2fa_required_{ig_user}" in st.session_state and st.session_state[f"2fa_required_{ig_user}"]:
                    if f"2fa_code_{ig_user}" not in st.session_state or not st.session_state[f"2fa_code_{ig_user}"]:
                        st.info("📱 Please complete the two-factor authentication above before posting.")
                    else:
                        # We have a 2FA code, attempt login
                        try:
                            cl = get_instagram_client()
                            login_result = cl.login(ig_user, ig_password)
                        
                            if login_result:
                                st.info("🔄 Uploading to Instagram...")
                            
                                img_data = requests.get(st.session_state.img_url).content
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                    tmp_file.write(img_data)
                                    tmp_file_path = tmp_file.name

                                cl.photo_upload(tmp_file_path, st.session_state.solution)
                                st.success("✅ Posted to Instagram successfully!")
                                os.remove(tmp_file_path)

                                # Reset after posting
                                st.session_state.ready_to_post = False
                            
                                # Clear any 2FA session state
                                for key in list(st.session_state.keys()):
                                    if key.startswith(f"2fa_") and ig_user in key:
                                        del st.session_state[key]
                                    
                        except TwoFactorRequired:
                            st.info("🔐 Two-factor authentication required. Please enter the code above.")
                        except Exception as e:
                            st.error(f"❌ Error posting to Instagram: {e}")
                else:
                    # First time login attempt
                    try:
                        cl = get_instagram_client()
                    
                        login_result = cl.login(ig_user, ig_password)
                    
                        if login_result:
                            st.info("🔄 Uploading to Instagram...")
                        
                            img_data = requests.get(st.session_state.img_url).content
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                tmp_file.write(img_data)
                                tmp_file_path = tmp_file.name

                            cl.photo_upload(tmp_file_path, st.session_state.solution)
                            st.success("✅ Posted to Instagram successfully!")
                            os.remove(tmp_file_path)

                            # Reset after posting
                            st.session_state.ready_to_post = False
                        
                    except TwoFactorRequired:
                        st.info("🔐 Two-factor authentication required. Please enter the code above and click 'Post to Instagram' again.")
                    except Exception as e:
                        st.error(f"❌ Error posting to Instagram: {e}")
    else:
        if st.session_state.already_requested_2fa:
            if not ig_user or not ig_password:
                st.error("Please enter Instagram credentials in the sidebar")
            else:
                # Check if 2FA is currently required and no code is available
                if f"2fa_required_{ig_user}" in st.session_state and st.session_state[f"2fa_required_{ig_user}"]:
                    if f"2fa_code_{ig_user}" not in st.session_state or not st.session_state[f"2fa_code_{ig_user}"]:
                        st.info("📱 Please complete the two-factor authentication above before posting.")
                    else:
                        # We have a 2FA code, attempt login
                        try:
                            cl = get_instagram_client()
                            login_result = cl.login(ig_user, ig_password)
                        
                            if login_result:
                                st.info("🔄 Uploading to Instagram...")
                            
                                img_data = requests.get(st.session_state.img_url).content
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                    tmp_file.write(img_data)
                                    tmp_file_path = tmp_file.name

                                cl.photo_upload(tmp_file_path, st.session_state.solution)
                                st.success("✅ Posted to Instagram successfully!")
                                os.remove(tmp_file_path)

                                # Reset after posting
                                st.session_state.ready_to_post = False
                            
                                # Clear any 2FA session state
                                for key in list(st.session_state.keys()):
                                    if key.startswith(f"2fa_") and ig_user in key:
                                        del st.session_state[key]
                                    
                        except TwoFactorRequired:
                            st.info("🔐 Two-factor authentication required. Please enter the code above.")
                        except Exception as e:
                            st.error(f"❌ Error posting to Instagram: {e}")
                else:
                    # First time login attempt
                    try:
                        cl = get_instagram_client()
                    
                        login_result = cl.login(ig_user, ig_password)
                    
                        if login_result:
                            st.info("🔄 Uploading to Instagram...")
                        
                            img_data = requests.get(st.session_state.img_url).content
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                tmp_file.write(img_data)
                                tmp_file_path = tmp_file.name

                            cl.photo_upload(tmp_file_path, st.session_state.solution)
                            st.success("✅ Posted to Instagram successfully!")
                            os.remove(tmp_file_path)

                            # Reset after posting
                            st.session_state.ready_to_post = False
                        
                    except TwoFactorRequired:
                        st.info("🔐 Two-factor authentication required. Please enter the code above and click 'Post to Instagram' again.")
                    except Exception as e:
                        st.error(f"❌ Error posting to Instagram: {e}")