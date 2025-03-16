all:
	python stocktrading.py
install:
	pip install -r requirements.txt
train:
	python3 model.py
collect:
	python3 collect_data.py
init:
	python3 data_management.py
rnn:
	python3 rnn.py
check:
	python3 crawler.py
