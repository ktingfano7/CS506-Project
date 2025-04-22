.PHONY: all install build run clean

all: install run

install:
	pip install -r requirements.txt
	pip install streamlit


build:
	@echo "No build step necessary."

run:
	streamlit run app.py

clean:
	rm -rf __pycache__ *.pyc