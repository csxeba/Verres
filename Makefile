include .env
export

run : build
	docker run -it --rm --network host \
    --name verres_environment \
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

tensorboard : build
	docker run --rm -d \
	--name verres_tensorboard \
	--hostname $(shell cat /etc/hostname) \
	-v ${host_artifactory_root}:/artifactory \
	-u $(shell id -u) \
	-p ${host_tensorboard_port}:6006 \
	trickster/environment:latest \
	tensorboard \
	--logdir /artifacotry \
	--bind_all

jupyter : build
	docker run --rm -d \
	--name verres_jupyter \
	--hostname $(shell cat /etc/hostname) \
	-v ${host_artifactory_root}:/artifactory \
	-v ${host_models_root}:/models \
	-v ${host_data_root}:/data \
	-v $(shell pwd):/workspace \
	-u $(shell id -u) \
	-p ${host_jupyter_port}:8888 \
	trickster/environment:latest \
	/opt/conda/bin/jupyter notebook \
	--no-browser \
	--port 8888 \
	--ip 0.0.0.0
	sleep 2
	docker logs verres_jupyter
