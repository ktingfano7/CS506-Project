.PHONY: install build run clean all


install:
	pip install streamlit


build:
	@echo "No build step necessary."

run:
	python recommender.py --dataset dataset.csv  --song "Shape of You"  --artist "Ed Sheeran"  --n_songs 5  --metric cosine  --same_genre

clean:
	rm -rf __pycache__ *.pyc

all: install run