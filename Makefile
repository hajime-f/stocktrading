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
clear:
	rm -rf stocklib/__pycache__
