all:
	python stocktrading.py
install:
	pip install -r requirements.txt
train:
	python model.py
collect:
	python collect_data.py
init:
	python data_management.py
rnn:
	python rnn.py
predict:
	python predictor.py
check:
	python crawler.py
