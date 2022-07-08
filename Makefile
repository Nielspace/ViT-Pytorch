setup:
	python3 -m venv ~/.flask-ml-azure-serverless
	#source ~/.streamlit-ml-azure/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

run:
	python app.py

all: install lint 