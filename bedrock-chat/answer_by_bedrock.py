import json, boto3, os
from langchain_community.llms.bedrock import Bedrock
from langchain/community.memory import ConversationBufferMemory
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chains import CoversationRetrievalChain

def main():
	bedrock_runtime = boto3.client(
		service_name = 'bedrock-runtime',
		region_name = 'us-west-2'
	)
