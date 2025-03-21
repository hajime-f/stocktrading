all:
	python stocktrading.py
install:
	pip install -r requirements.txt
collect:
	python collect_data.py
init:
	python data_management.py
update:
	python update_model.py
predict:
	python predictor_2.py
check:
	python crawler.py
check2:
	python crawler2.py
