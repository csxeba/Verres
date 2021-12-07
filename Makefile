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
	-e DISPLAY \
	-v /tmp/.X11-unix/:/tmp/.X11-unix \
	-v /home/$${USER}/.Xauthority:/home/$${USER}/.Xauthority \
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
	--build-arg ssh_password=${VERRES_SSH_PASSWORD} \
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
	verres/environment:latest \
	tensorboard \
	--logdir /artifactory \
	--bind_all
	sleep 2
	docker logs verres_tensorboard

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
	verres/environment:latest \
	/opt/conda/bin/jupyter notebook \
	--no-browser \
	--port 8888 \
	--ip 0.0.0.0
	sleep 2
	docker logs verres_jupyter

run-ssh : environment_setup_sample.sh
	docker run -it --rm \
	--name verres_shh_server \
	--hostname $(shell cat /etc/hostname) \
	-v ${VERRES_ARTIFACTORY}:/artifactory \
	-v ${VERRES_MODELS}:/models \
	-v ${VERRES_DATA}:/data \
	-p ${VERRES_SSH_PORT}:22 \
	verres/environment:latest

clean :
	-docker kill verres_environment
	-docker kill verres_tensorboard
	-docker kill verres_jupyter
	-docker kill verres_ssh_server
	-docker rmi verres/environment:latest
	docker system prune -f
