all:
	docker exec -it python3 python stocktrading.py
build:
	docker-compose build --no-cache
install:
	docker-compose build
up:
	docker-compose up -d
ps:
	docker-compose ps
version:
	docker exec -it python3 python --version
down:
	docker-compose down
bash:
	docker-compose exec python3 bash
ls:
	docker container ls
