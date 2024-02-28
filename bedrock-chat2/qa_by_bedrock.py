import boto3
import qdrant_client

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Qdrant

QDRANT_PATH = "/home/rocky/aichat/bedrock-chat2/local_qdrant"
COLLECTION_NAME = "amazon-titan-embed-collection"  # Qdrantに保存されるコレクションの名前
REGION_NAME = "us-west-2"  # 使用するAWSリージョンの名前
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # テキスト埋め込みモデルのID
CHAT_MODEL_ID = "anthropic.claude-instant-v1"
QUERY = "Amazon FSx for NetApp ONTAP の特徴を教えて"

def generate_prompt():
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
	llm = Bedrock(
		client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),
		model_id = CHAT_MODEL_ID,
		verbose = True
	)

	return llm

def build_qa_model(prompts, llm):
	client = qdrant_client.QdrantClient(path=QDRANT_PATH)
	# collections = client.get_collections().collections
	# collection_names = [collection.name for collection in collections]
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

def ask(qa):
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
