import boto3
import qdrant_client
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import BedrockEmbeddings

QDRANT_PATH = "/home/rocky/aichat/bedrock-chat2/local_qdrant"
COLLECTION_NAME = "amazon-titan-embed-collection"  # Qdrantに保存されるコレクションの名前
REGION_NAME = "us-west-2"  # 使用するAWSリージョンの名前
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"  # テキスト埋め込みモデルのID

client = qdrant_client.QdrantClient(path=QDRANT_PATH)  # Qdrantクライアントを作成
qdrant = Qdrant(
	client = client,
	collection_name = COLLECTION_NAME,  # コレクション名を指定
	embeddings = BedrockEmbeddings(
		client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME),  # Bedrockランタイムクライアントを作成
		model_id = EMBEDDING_MODEL_ID  # 埋め込みモデルのIDを指定
	)
)

query = "Amazon FSx for NetApp ONTAP の特徴を教えてください"  # 検索クエリを設定

docs = qdrant.similarity_search(query=query, k=4)  # 類似度検索を実行し、上位4件を取得

for i in docs:
	print({"content": i.page_content, "metadata": i.metadata})  # 検索結果を出力
