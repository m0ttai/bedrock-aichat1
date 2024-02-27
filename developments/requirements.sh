#! /bin/bash

# get-pip.pyスクリプトをダウンロードして、Pythonで実行する
curl -kL https://bootstrap.pypa.io/get-pip.py | python

# Pythonのpipモジュールを使用して、複数のパッケージを一括でインストールする
python -m pip install boto3 streamlit streamlit_chat langchain langchain-community bs4 PyPDF openai faiss-cpu
