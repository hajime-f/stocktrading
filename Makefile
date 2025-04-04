all:
	python stocktrading.py
install:
	pip install -r requirements.txt
init:
	python data_manager.py
predict:
	python predictor.py
check:
	python crawler.py
