.PHONY: install build run clean all


install:
	$(PYTHON) -m pip install --upgrade pip
	pip install pandas numpy scikit-learn streamlit
	pip install streamlit
	pip install jupyter
	pip install kagglehub
	pip install pywin32


build:
	jupyter nbconvert --to notebook --execute --inplace dataprocess.ipynb

test:
	python recommender.py --dataset dataset.csv  --song "Shape of You"  --artist "Ed Sheeran"  --n_songs 5  --metric cosine  --same_genre

	python recommender.py --dataset dataset.csv  --song "Love Me Like You Do"  --artist "Ellie Goulding"  --n_songs 10  --metric euclidean  --same_genre

	python recommender.py --dataset dataset.csv  --song "I Ain't Worried"  --artist "OneRepublic"  --n_songs 15  --metric manhattan

clean:
	rm -rf __pycache__ *.pyc

run:
	streamlit run app.py

all: install build