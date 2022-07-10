setup:
	python3 -m venv ~/.visiontransformer
	source ~/.visiontransformer/bin/activate
	cd .visiontransformer

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

run:
	python source/test.py

all: install run 