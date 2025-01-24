from langchain_experimental.tools import PythonAstREPLTool
from langchain_teddynote import logging
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from utils import create_agent, validate_api_key, tool_callback, observation_callback, result_callback, print_messages, ask
 
# API 키 및 프로젝트 설정
load_dotenv()
logging.langsmith("CSV Agent 챗봇")


# Streamlit 페이지 설정 및 스타일
st.set_page_config(
    page_title="CSV 데이터 분석 챗봇",
    page_icon="💬",
)

# 커스텀 CSS 스타일
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stFileUploader > div > div {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stChat > div {
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.title("CSV 데이터 분석 챗봇 💬")
st.markdown("---")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []



# 사이드바 설정을 API 키 입력 부분만 수정
with st.sidebar:
    st.markdown("### 🔑 설정")
    api_key = st.text_input("OpenAI API 키를 입력해주세요", type="password")
    
    if api_key:
        if validate_api_key(api_key):
            st.success("API 키가 유효합니다! ✅")
            st.session_state['api_key_valid'] = True
        else:
            st.error("유효하지 않은 API 키입니다. 다시 확인해주세요.")
            st.session_state['api_key_valid'] = False
    
    st.markdown("### 🔄 대화 관리")
    clear_btn = st.button("대화 초기화", use_container_width=True)
    
    st.markdown("### 📁 파일 업로드")
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드 해주세요.",
        type=["csv"],
        help="분석하고자 하는 CSV 파일을 이곳에 드래그앤드롭 하거나 선택해주세요."
    )
    
    st.markdown("### 🤖 모델 설정")
    selected_model = st.selectbox(
        "OpenAI 모델을 선택해주세요.",
        ["gpt-4o", "gpt-4o-mini"],
        index=0
    )
    
    apply_btn = st.button("데이터 분석 시작", use_container_width=True)



# 메인 로직 수정
if clear_btn:
    st.session_state["messages"] = []

if apply_btn:
    if not st.session_state.get('api_key_valid', False):
        st.error("유효한 API 키를 먼저 입력해주세요.")
    elif not uploaded_file:
        st.warning("파일을 업로드 해주세요.")
    else:
        with st.spinner("데이터를 불러오는 중..."):
            loaded_data = pd.read_csv(uploaded_file)
            st.session_state["df"] = loaded_data
            st.session_state["python_tool"] = PythonAstREPLTool()
            st.session_state["python_tool"].locals["df"] = loaded_data
            st.session_state["agent"] = create_agent(loaded_data, selected_model)
            st.success("✨ 설정이 완료되었습니다. 대화를 시작해 주세요!")

print_messages()  # 저장된 메시지 출력

user_input = st.chat_input("궁금한 내용을 물어보세요!")  # 사용자 입력 받기
if user_input:
    ask(user_input)  # 사용자 질문 처리