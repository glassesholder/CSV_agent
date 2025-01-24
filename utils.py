import openai 
import streamlit as st
from typing import List, Union
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import time

i = 0

# OpenAI API 키 검증 함수
def validate_api_key(api_key: str) -> bool:
    """OpenAI API 키의 유효성을 검증합니다."""
    if not api_key:
        return False
    
    try:
        openai.api_key = api_key
        # 간단한 API 호출로 키 검증
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=3
        )
        return True
    except Exception as e:
        return False
    
# 상수 정의
class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = "user"  # 사용자 메시지 역할
    ASSISTANT = "assistant"  # 어시스턴트 메시지 역할


class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = "text"  # 텍스트 메시지
    FIGURE = "figure"  # 그림 메시지
    CODE = "code"  # 코드 메시지
    DATAFRAME = "dataframe"  # 데이터프레임 메시지
    PLOTLY = "plotly"  # Plotly 그래프 메시지

# 메시지 관련 함수
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """
    for role, content_list in st.session_state["messages"]:
        with st.chat_message(role):
            for content in content_list:
                if isinstance(content, list):
                    global i
                    message_type, message_content = content

                    if message_type == MessageType.TEXT:
                        st.markdown(message_content)  # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:
                        st.pyplot(message_content)  # 그림 메시지 출력
                    elif message_type == MessageType.CODE:
                        with st.status("코드 출력", expanded=False):
                            st.code(
                                message_content, language="python"
                            )  # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:
                        st.dataframe(message_content)  # 데이터프레임 메시지 출력
                    elif message_type == MessageType.PLOTLY:  # Plotly 그래프 처리
                        print('이전 plotly 그래프를 잘 출력했습니다.')
                        i += 1
                        st.plotly_chart(message_content, use_container_width=True, key=f"plot_{i}")
                else:
                    raise ValueError(f"알 수 없는 콘텐츠 유형: {content}")


def add_message(role: MessageRole, content: List[Union[MessageType, str]]):
    """
    새로운 메시지를 저장하는 함수입니다.

    Args:
        role (MessageRole): 메시지 역할 (사용자 또는 어시스턴트)
        content (List[Union[MessageType, str]]): 메시지 내용
    """
    messages = st.session_state["messages"]
    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])  # 같은 역할의 연속된 메시지는 하나로 합칩니다
    else:
        messages.append([role, [content]])  # 새로운 역할의 메시지는 새로 추가합니다



# 콜백 함수
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.
    """
    if tool_name := tool.get("tool"):
        if tool_name == "python_repl_ast":
            tool_input = tool.get("tool_input", {})
            query = tool_input.get("query")
            if query:
                df_in_result = None
                fig_output = None
                with st.status("데이터 분석 중...", expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")
                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])
                    if "df" in st.session_state:
                        result = st.session_state["python_tool"].invoke(
                            {"query": query}
                        )
                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                        elif isinstance(result, (go.Figure, dict)):
                            fig_output = result
                    status.update(label="코드 출력", state="complete", expanded=False)

                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(
                        MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result]
                    )

                if fig_output is not None:
                    global i
                    if isinstance(fig_output, dict) and 'data' in fig_output:
                        print('plotly 그래프 처리 방식 1')
                        i += 1
                        fig = go.Figure(fig_output)
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
                        add_message(MessageRole.ASSISTANT, [MessageType.PLOTLY, fig])
                    if isinstance(fig_output, go.Figure):
                        print('polotly 그래프 처리 방식 2')
                        i += 1
                        st.plotly_chart(fig_output, use_container_width=True, key=f"plot_{i}")
                        add_message(MessageRole.ASSISTANT, [MessageType.PLOTLY, fig_output])
                        print('plotly 그래프가 잘 추가 되었습니다.')

                if "plt.show" in query:
                    print('plt 그래프 처리 방식 1')
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                return result  # 이 부분이 중요합니다
            else:
                st.error("데이터프레임이 정의되지 않았습니다. CSV 파일을 먼저 업로드해주세요.")
                return


def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict): 관찰 결과
    """
    if "observation" in observation:
        obs = observation["observation"]
        if isinstance(obs, str) and "Error" in obs:
            st.error(obs)
            st.session_state["messages"][-1][
                1
            ].clear()  # 에러 발생 시 마지막 메시지 삭제


def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str): 최종 결과
    """
    pass  # 현재는 아무 동작도 하지 않습니다


# 에이전트 생성 함수
def create_agent(dataframe, selected_model="gpt-4o"):
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Args:
        dataframe (pd.DataFrame): 분석할 데이터프레임
        selected_model (str, optional): 사용할 OpenAI 모델. 기본값은 "gpt-4o"

    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """
    return create_pandas_dataframe_agent(
        ChatOpenAI(model=selected_model, temperature=0),
        dataframe,
        verbose=False,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request.\n"
        "Do not contain image path like '![Survival Rate by Class and Sex](data:image/png;base64,iVBORw0KGgoAAAANSUh' in your answer"
        "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use plotly for interactive visualization.\n"
        "For plotly visualizations, follow these guidelines: \n" 
        "- Return the figure object directly without using .show().\n"
        "For matplotlib/seaborn visualizations, follow these guidelines: \n"
        "- Use plt.show() at the end of your code\n"
        "- When using seaborn palette without hue, assign x variable to hue and set legend=False."
        "\n\n<Visualization Preference>\n"
        "[IMPORTANT] Use `English` for your visualization title and labels.\n"
        "- For plotly: Use a clean white background and professional color schemes.\n"
        "- For matplotlib/seaborn: Use `muted` cmap, white background, and no grid."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "\n\n[IMPORTANT] The language of final answer should be written in Korean with your enlightenment.\n"
        "Please answer final answer within 300 characters. "
        "'보여줘' does not mean 'give me python code'.\n"
        "In any question, you should not provide python code in your final answer.\n"
        "Under no circumstances should the image path be provided like '![Survival Rate by Class and Sex](data:image/png;base64,iVBORw0KGgoAAAANSUh'"
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below."
    )


# 질문 처리 함수
def ask(query):
    """
    사용자의 질문을 처리하고 응답을 생성하는 함수입니다.
    """
    if "agent" in st.session_state:
        st.chat_message("user").write(query)
        print('사용자 질문 : ' + query)
        add_message(MessageRole.USER, [MessageType.TEXT, query])

        agent = st.session_state["agent"]
        print('에이전트 확인 : ' + str(agent))
        response = agent.stream({"input": query})
        print('응답 생성을 위한 작업 1 : ' + str(response))

        ai_answer = ""
        parser_callback = AgentCallbacks(
            tool_callback, observation_callback, result_callback
        )
        stream_parser = AgentStreamParser(parser_callback)
        print('StreamParser : ' + str(stream_parser))

        with st.chat_message("assistant"):
            for step in response:
                print('응답 생성 단계 시작')
                time.sleep(1)
                stream_parser.process_agent_steps(step)
                print('응답 생성 단계 완료')
                if "output" in step:
                    ai_answer += step["output"]
                    st.write(ai_answer)  # 각 스텝마다 답변을 업데이트합니다

        if ai_answer:  # 답변이 있을 경우에만 메시지에 추가
            print('최종 응답')
            add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])