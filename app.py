from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# 環境変数からOpenAI APIキーを読み込む
# Streamlit Community Cloudにデプロイする際は、Settings > Secrets に OPENAI_API_KEY を設定してください
# ローカルで実行する場合は、.envファイルなどから読み込むか、直接 st.secrets["OPENAI_API_KEY"] を使います
# 例: os.environ["OPENAI_API_KEY"] = "sk-..."
# ...existing code...
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI APIキーが設定されていません。環境変数または.envファイルにOPENAI_API_KEYを設定してください。")
    st.stop()
# ...existing code...


def generate_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類を受け取り、LLMからの回答を返す関数。
    """
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key) # 最新のモデル名を使用

    # 専門家の種類に応じてシステムメッセージを変更
    if expert_type == "ITコンサルタント":
        system_message = "あなたはITコンサルタントです。ユーザーのITに関する課題に対して、具体的かつ実践的な解決策を提案してください。"
    elif expert_type == "キャリアアドバイザー":
        system_message = "あなたはキャリアアドバイザーです。ユーザーのキャリアに関する悩みや目標に対して、的確なアドバイスと具体的な行動計画を提示してください。"
    else:
        system_message = "あなたは親切なアシスタントです。ユーザーの質問に何でも答えてください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{input}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke({"input": user_input})
    return response

# StreamlitアプリケーションのUI
st.set_page_config(page_title="専門家AIチャットアプリ", page_icon="🤖")

st.title("🤖 専門家AIチャットアプリ")

st.markdown(
    """
    このアプリは、あなたの質問に対して、選択した専門家が回答します。
    以下の手順でご利用ください。

    1. **質問を入力してください。**
    2. **回答してほしい専門家をラジオボタンで選択してください。**
    3. **「回答を生成」ボタンをクリックすると、LLMが回答を生成します。**
    """
)

# 入力フォーム
user_input = st.text_area("ここに質問を入力してください:", height=150)

# ラジオボタンで専門家を選択
expert_type = st.radio(
    "LLMに振る舞わせる専門家を選択してください:",
    ("ITコンサルタント", "キャリアアドバイザー", "一般的なアシスタント"),
    index=0 # デフォルトで「ITコンサルタント」を選択
)

if st.button("回答を生成"):
    if user_input:
        with st.spinner("LLMが回答を生成中です..."):
            response = generate_llm_response(user_input, expert_type)
            st.subheader("回答:")
            st.write(response)
    else:
        st.warning("質問を入力してください。")

st.markdown(
    """
    ---
    ### 補足
    - OpenAI APIを利用しているため、ご利用にはAPIキーが必要です。
    - Streamlit Community Cloudにデプロイする際は、Pythonのバージョンを3.11に設定してください。
    """
)