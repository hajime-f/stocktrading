all:
	python stocktrading.py
install:
	pip install -r requirements.txt
collect:
	python collect_data.py
init:
	python data_management.py
update2:
	python update_model_2.py
predict2:
	python predictor_2.py
check:
	python crawler.py
check2:
	python crawler_2.py
sim:
	python simulator.py
