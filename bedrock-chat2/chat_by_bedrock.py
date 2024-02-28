import boto3
import qdrant_client
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

QDRANT_PATH = "/home/rocky/aichat/bedrock-chat2/local_qdrant"
COLLECTION_NAME = "amazon-titan-embed-collection"  # Qdrantに保存されるコレクションの名前
REGION_NAME = "us-west-2"  # 使用するAWSリージョンの名前
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # テキスト埋め込みモデルのID
BEDROCK_CHAT_MODEL_ID = "anthropic.claude-instant-v1"
OPENAI_CHAT_MODEL_ID = "gpt-3.5-turbo"


##### Back-End #####
def generate_prompt():
	prompt_template = '''Human:
	Text: {context}

	Question: {question}

	Answer in Japanese within the 200 characters, and be sure to add "AIでございます！" at the beginning of your answer.

	Assistant:
	'''

	prompts = PromptTemplate(
		template = prompt_template,
		input_variables = ['context', 'question']
	)

	return prompts

def call_llm_bedrock(model_id):
	llm = Bedrock(
		client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),
		model_id = model_id,
		verbose = True
	)

	return llm

def build_qa_model(prompts, llm):
	client = qdrant_client.QdrantClient(path=QDRANT_PATH)
	qdrant = Qdrant(
		client = client,
		collection_name = COLLECTION_NAME,  # コレクション名を指定
		embeddings = BedrockEmbeddings(
			client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),  # Bedrockランタイムクライアントを作成
			model_id = EMBEDDING_MODEL_ID  # 埋め込みモデルのIDを指定
		)
	)

	retriever = qdrant.as_retriever(
		search_type="similarity",
		search_kwargs={"k":4}
	)

	chain_type_kwargs = {"prompt": prompts}

	qa = RetrievalQA.from_chain_type(
		llm = llm,
		chain_type = "stuff",
		retriever = retriever,
		chain_type_kwargs = chain_type_kwargs,
		return_source_documents = False
	)

	return qa

def ask(qa, messages):
	answer = qa(messages)

	return answer


##### Front-End #####
def init_page():
	st.set_page_config(
		page_title = "Hello RAG World !!!"
	)
	st.header("Hello RAG World!!!")
	st.sidebar.title("<Optimize AI Menu>")

def init_messages():
	clear_button = st.sidebar.button("Clear", key="clear")
	if clear_button or "messages" not in st.session_state:
		st.session_state.messages = [SystemMessage(content="Please Give me orders.")]

def select_model():
	temperature = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

	selected_model = st.sidebar.radio("Select Model", ("OpenAI", "Claude"))
	if selected_model == "OpenAI":
		llm = ChatOpenAI(temperature=temperature, model_name=OPENAI_CHAT_MODEL_ID)
	else:
		llm = call_llm_bedrock(BEDROCK_CHAT_MODEL_ID)

	return llm


##### Main #####
def main():
	init_page()

	prompts = generate_prompt()
	llm = select_model()
	qa = build_qa_model(prompts, llm)

	init_messages()

	if user_input := st.chat_input("Please Give me orders."):
		st.session_state.messages.append(HumanMessage(content=user_input))
		with st.spinner("AI is typing..."):
			answer = ask(qa, user_input)
		st.session_state.messages.append(AIMessage(content=answer["result"]))

	# messages = st.session_state.messages
	messages = st.session_state.get("messages", [])
	for message in messages:
		if isinstance(message, AIMessage):
			with st.chat_message('assistant'):
				st.markdown(message.content)
		elif isinstance(message, HumanMessage):
			with st.chat_message('user'):
				st.markdown(message.content)
		else:  # isinstance(message, SystemMessage):
			st.write(f"System message: {message.content}")

if __name__ == '__main__':
    main()
