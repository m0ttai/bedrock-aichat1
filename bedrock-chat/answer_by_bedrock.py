import json, boto3, os
from langchain_community.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def main():
	bedrock_runtime = boto3.client(
		service_name = 'bedrock-runtime',
		region_name = 'us-west-2'
	)

	embeddings = BedrockEmbeddings(
		model_id = 'amazon.titan-embed-text-v1',
        client = bedrock_runtime,
        region_name = 'us-west-2'
	)

	llm = Bedrock(
		model_id = 'anthropic.claude-v2',
		client = bedrock_runtime,
        region_name = 'us-west-2'
	)

	faiss_index = FAISS.load_local('/home/rocky/aichat/bedrock-chat', embeddings)

	qa = ConversationalRetrievalChain.from_llm(
		llm = llm,
		retriever = faiss_index.as_retriever(),
		return_source_document = True
	)

	human_input = "FSx for NetApp ONTAP とは何ですか？"

	res = qa({"question": human_input})

	return(json.dumps(res["answer"]))

if __name__ == '__main__':
	main()
