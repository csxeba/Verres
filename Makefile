include .env
export

run : build
	docker run -it --rm --network host \
	-v ${host_artifactory_root}:/artifactory \
	-v ${host_models_root}:/models \
	-v ${host_data_root}:/data \
	-v $(shell pwd):/workspace \
	-u $(shell id -u) \
	trickster/environment:latest \
	/bin/bash

build :
	docker build \
	--tag trickster/environment:latest \
	--network host \
	--build-arg uid=$(shell id -u) \
	--build-arg accelerator=${compute_accelerator} \
	-f docker/Dockerfile \
	.
