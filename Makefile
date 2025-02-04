all:
	python3 stocktrading.py
install:
	pip3 install -r requirements.txt
train:
	python3 train_model.py
data:
	python3 collect_data.py
clear:
	rm -rf stocklib/__pycache__
