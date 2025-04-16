all:
	python stocktrading.py
install:
	pip install -r requirements.txt
init:
	python data_manager.py
update:
	python updater.py
check:
	python crawler.py
