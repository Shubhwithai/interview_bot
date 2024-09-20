import streamlit as st
from streamlit_chat import message
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# LLM and memory setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
memory = ConversationBufferMemory(return_messages=True)

class CharCreationChain(ConversationChain):
    @classmethod
    def from_description(cls, description):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(description),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return cls(prompt=prompt, llm=llm, memory=memory)

# Interviewer personas and details
char_descriptions = {
    "pm_interviewer_1": "You are Vivian Reyes, a sharp-witted product management guru...",
    "pm_interviewer_2": "You are William Thompson, a seasoned product management expert...",
    "pm_interviewer_3": "You are Dr. Priya Nair, a data-driven product leader..."
}

interviewer_details = {
    "pm_interviewer_1": {"name": "Vivian Reyes", "role": "Director of Product", "company": "Google"},
    "pm_interviewer_2": {"name": "William Thompson", "role": "Senior Product Manager", "company": "Airbnb"},
    "pm_interviewer_3": {"name": "Dr. Priya Nair", "role": "CPO", "company": "Scale AI"}
}

# Helper functions
def format_interviewer_option(option):
    details = interviewer_details[option]
    return f"{details['name']}, {details['role']} at {details['company']}"

def remove_last_name(full_name):
    return ' '.join(full_name.split()[:-1])

def get_initial_message():
    return [
        {"role": "user", "content": "Can you interview me for the PM role?"},
        {"role": "assistant", "content": "Of course! Let's begin."}
    ]

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

# Streamlit app
st.set_page_config(page_title="Interview Bot", page_icon="🤖", layout="wide")

st.title("Interview Bot 🤖")

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

# User preferences form
if not st.session_state.submitted:
    st.write("🚀 This app lets you engage in lifelike interview simulations, helping you build your confidence and skills.")
    st.write("👉 To get started, choose your interviewer, role, and topic from the form below. Then, click on the button to begin your interview!")

    with st.form("preferences"):
        st.write("Choose your parameters")
        st.session_state.user_name = st.text_input("Enter your name:")
        st.session_state.interviewer = st.selectbox("Choose your interviewer", 
                                                    options=list(interviewer_details.keys()), 
                                                    format_func=format_interviewer_option)
        st.session_state.role = st.selectbox("Choose your role:", ["APM", "PM", "Senior PM"])
        st.session_state.topic = st.selectbox("Choose your topic:", ["Product Strategy", "User Research", "Feature Development"])
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            st.session_state.submitted = True

# Interview session
if st.session_state.submitted:
    st.write(f"Hello, {st.session_state.user_name}!")
    interviewer = interviewer_details[st.session_state.interviewer]
    st.markdown(f"""
    - You will now be interviewed by {interviewer['name']}, {interviewer['role']} at {interviewer['company']}.
    - You will be asked questions as per the {st.session_state.role} role and the questions will be based on {st.session_state.topic}.
    - Feel free to ask questions and engage in a conversation with {remove_last_name(interviewer['name'])}.
    """)

    if st.button("Begin Interview"):
        st.session_state.interview_started = True

    if st.session_state.interview_started:
        conversation = CharCreationChain.from_description(char_descriptions[st.session_state.interviewer])

        if "messages" not in st.session_state:
            st.session_state.messages = get_initial_message()
            initial_question_prompt = f"Prepare a question on {st.session_state.topic} for a {st.session_state.role} role."
            initial_question = conversation.predict(input=initial_question_prompt)
            st.session_state.messages = update_chat(st.session_state.messages, "assistant", initial_question)

        # Display chat messages
        for msg in st.session_state.messages[1:]:  # Skip the initial system message
            message(msg["content"], is_user=msg["role"] == "user", key=str(hash(msg["content"])))

        # User input
        user_input = st.chat_input("Your response:")
        if user_input:
            st.session_state.messages = update_chat(st.session_state.messages, "user", user_input)
            with st.spinner("Thinking..."):
                response = conversation.predict(input=user_input)
                st.session_state.messages = update_chat(st.session_state.messages, "assistant", response)
            st.rerun()

# Sidebar with interview information
with st.sidebar:
    st.header("Interview Information")
    if st.session_state.submitted:
        st.write(f"**Interviewee:** {st.session_state.user_name}")
        st.write(f"**Interviewer:** {interviewer_details[st.session_state.interviewer]['name']}")
        st.write(f"**Role:** {st.session_state.role}")
        st.write(f"**Topic:** {st.session_state.topic}")
    else:
        st.write("Please submit your preferences to start the interview.")