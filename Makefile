setup:
	python3 -m venv ~/.streamlit-ml-azure-serverless
	#source ~/.streamlit-ml-azure-serverless/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# run:
# 	python source/test.py

all: install  