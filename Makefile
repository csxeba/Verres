include environment_setup.sh
export

run-env : environment_setup.sh
	docker run -it --rm --network host \
    --name verres_environment \
	-v $(realpath ${VERRES_ARTIFACTORY}):/artifactory \
	-v $(realpath ${VERRES_MODELS}):/models \
	-v $(realpath ${VERRES_DATA}):/data \
	-v $(shell pwd):/workspace \
	-u $(shell id -u) \
	verres/environment:latest \
	/bin/bash

run : run-env

build-env : environment_setup.sh
	docker build \
	--tag verres/environment:latest \
	--network host \
	--build-arg uid=$(shell id -u) \
	--build-arg username=$${USER} \
	--build-arg accelerator=${VERRES_ACCELERATOR} \
	--build-arg http_proxy=${http_proxy} \
	--build-arg https_proxy=${https_proxy} \
	-f docker/Dockerfile \
	.

build-environment: build-env

build : build-env

run-tensorboard : environment_setup.sh
	docker run --rm -d \
	--name verres_tensorboard \
	--hostname $(shell cat /etc/hostname) \
	-v ${VERRES_ARTIFACTORY}:/artifactory \
	-u $(shell id -u) \
	-p ${VERRES_TENSORBOARD_PORT}:6006 \
	trickster/environment:latest \
	tensorboard \
	--logdir /artifacotry \
	--bind_all

run-jupyter : environment_setup_sample.sh
	docker run --rm -d \
	--name verres_jupyter \
	--hostname $(shell cat /etc/hostname) \
	-v ${VERRES_ARTIFACTORY}:/artifactory \
	-v ${VERRES_MODELS}:/models \
	-v ${VERRES_DATA}:/data \
	-v $(shell pwd):/workspace \
	-u $(shell id -u) \
	-p ${VERRES_JUPYTER_PORT}:8888 \
	trickster/environment:latest \
	/opt/conda/bin/jupyter notebook \
	--no-browser \
	--port 8888 \
	--ip 0.0.0.0
	sleep 2
	docker logs verres_jupyter

clean :
	-docker kill verres_environment
	-docker kill verres_tensorboard
	-docker kill verres_jupyter > /dev/null
	-docker rmi verres/environment:latest
	docker system prune -f
