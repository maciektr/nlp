elastic_build:
	docker network create elastic
	docker build --tag elastic_morfologik .
elastic_run:
	docker run --name elasticsearch --net elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -t elastic_morfologik