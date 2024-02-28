import boto3
import streamlit as st

from PyPDF2 import PdfReader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
QDRANT_PATH = "/home/rocky/aichat/bedrock-chat2/local_qdrant"
COLLECTION_NAME = "amazon-titan-embed-collection"  # Qdrantに保存されるコレクションの名前
REGION_NAME = "us-west-2"  # 使用するAWSリージョンの名前
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # テキスト埋め込みモデルのID
CHAT_MODEL_ID = "anthropic.claude-instant-v1"
QUERY = "Amazon FSx for NetApp ONTAP の特徴を教えて"

def generate_prompt():
	# HumanとAssistantがやり取りするプロンプトのテンプレートを定義
	prompt_template = '''Human:
	Text: {context}

	Question: {question}

	Answer in English the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available.

	Assistant:
	'''

	prompts = PromptTemplate(
		template = prompt_template,
		input_variables = ['context', 'question']
	)

	return prompts

def call_llm():
	# Bedrockクラスのインスタンスを作成して返す関数
	llm = Bedrock(
		client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),
		model_id = CHAT_MODEL_ID,
		verbose = True
	)

	return llm

def build_qa_model(prompts, llm):
	# QdrantとBedrockを組み合わせたQAモデルを構築する関数
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
		search_type = "similarity",
		search_kwargs = {"k": 4}
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

def ask(qa):
	# QAモデルにクエリを投げて回答を取得する関数
	answer = qa(QUERY)

	return answer

def main():
	prompts = generate_prompt()
	llm = call_llm()
	qa = build_qa_model(prompts, llm)
	answer = ask(qa)
	print(answer["result"])

if __name__ == '__main__':
	main()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "/home/rocky/aichat/bedrock-chat2/local_qdrant"
COLLECTION_NAME = "amazon-titan-embed-collection"  # Qdrantに保存されるコレクションの名前
REGION_NAME = "us-west-2"  # 使用するAWSリージョンの名前
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # テキスト埋め込みモデルのID

def get_pdf_text():
	pdf_reader = PdfReader('/home/rocky/aichat/bedrock-chat2/fsxn.pdf')  # PDFファイルを読み込む
	text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])  # 各ページからテキストを抽出して連結
	text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
		chunk_size = 250,  # テキストを分割するサイズ
		chunk_overlap = 0  # 分割時のオーバーラップ
	)
	return text_splitter.split_text(text)  # テキストを指定されたサイズで分割して返す

def load_qdrant():
	client = QdrantClient(path=QDRANT_PATH)  # Qdrantクライアントを作成

	collections = client.get_collections().collections  # Qdrantに存在するコレクションを取得
	collection_names = [collection.name for collection in collections]  # コレクション名のリストを作成

	if COLLECTION_NAME not in collection_names:
		client.create_collection(
			collection_name = COLLECTION_NAME,  # 新しいコレクションを作成
			vectors_config = VectorParams(size=1536, distance=Distance.COSINE),  # ベクトル設定を指定
		)
		print('collection created')  # コレクションが作成されたことを表示

	return Qdrant(
		client = client,
		collection_name = COLLECTION_NAME,
		embeddings = BedrockEmbeddings(
			client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),  # Bedrockランタイムクライアントを作成
			model_id = EMBEDDING_MODEL_ID  # 埋め込みモデルのIDを指定
		)
	)

def build_vector_store(pdf_text):
	qdrant = load_qdrant()  # Qdrantをロード
	qdrant.add_texts(pdf_text)  # テキストをQdrantに追加

def main():
	pdf_text = get_pdf_text()  # PDFからテキストを取得
	build_vector_store(pdf_text)  # ベクトルストアを構築

if __name__ == '__main__':
	main()
