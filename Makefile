.PHONY: up down logs ps pg redis minio kafka-topics clean

up:
	docker compose up -d

down:
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps

pg:
	docker exec -it ghostsig-postgres-1 psql -U ghostsig -d ghostsig

redis:
	docker exec -it ghostsig-redis-1 redis-cli

minio:
	@echo "MinIO Console: http://localhost:9001 (ghostsig / ghostsig_dev)"

kafka-topics:
	docker exec -it ghostsig-kafka-1 kafka-topics --list --bootstrap-server localhost:9092

clean:
	docker compose down -v && rm -rf pgdata miniodata
