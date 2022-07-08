setup:
	python3 -m venv ~/.flask-ml-azure-serverless
	#source ~/.streamlit-ml-azure/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

run:
	streamlit run source/app.py

all: install  