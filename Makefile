all:
	python stocktrading.py
install:
	pip install -r requirements.txt
init:
	python data_manager.py
predict:
	python model_manager.py
check:
	python crawler.py
