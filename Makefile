setup:
	python3 -m venv ~/.flask-ml-azure-serverless
	#source ~/.streamlit-ml-azure/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,W1203,W0702 app.py

run:
	python app.py

all: install lint test