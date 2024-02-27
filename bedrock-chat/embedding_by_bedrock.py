import json, boto3, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS

def main():
	# PyPDFLoaderを使用して指定されたPDFファイルをロードする
	loader = PyPDFLoader("/home/rocky/aichat/bedrock-chat/fsxn.pdf")

	# bedrock-runtimeとの接続を確立する
	bedrock_runtime = boto3.client(
		service_name='bedrock-runtime',
		region_name='us-west-2'
	)

	# BedrockEmbeddingsを初期化する
	embeddings = BedrockEmbeddings(
		model_id = 'amazon.titan-embed-text-v1',
		client = bedrock_runtime,
		region_name = 'us-west-2'
	)

	# FAISSを使用してベクトルストアのインデックス作成を行う
	index_creator = VectorstoreIndexCreator(
		vectorstore_cls = FAISS,
		embedding = embeddings
	)

	# ローダーからインデックスを作成し、ローカルに保存する
	index_from_loader = index_creator.from_loaders([loader])
	index_from_loader.vectorstore.save_local('/home/rocky/aichat/bedrock-chat')


if __name__ == '__main__':
	main()
