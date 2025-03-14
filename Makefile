all:
	python3 stocktrading.py
install:
	pip3 install -r requirements.txt
train:
	python3 train_model.py
collect:
	python3 collect_data.py
test:
	python3 test.py
init:
	python3 data_management.py
lstm:
	python3 lstm.py
backtest:
	python3 backtest.py
check:
	python3 crawler.py
clear:
	rm -rf stocklib/__pycache__
