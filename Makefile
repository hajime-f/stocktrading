all:
	python3 stocktrading.py
install:
	pip3 install -r requirements.txt
clear:
	rm -rf stocklib/__pycache__
