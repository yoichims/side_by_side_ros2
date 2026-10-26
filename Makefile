ROS2_DISTRO         ?= $(shell cat .env | grep IMAGE_TAG | cut -d = -f 2)
IMAGE_NAME          := sbs_devel
IMAGE_TAG           = $(ROS2_DISTRO)
APT_MIRROR          ?= jp
USER_ID             :=$(shell id -u)
GROUP_ID            :=$(shell id -g)
# Host gid of the DRI render nodes, so the group has a name inside the image.
# Empty on a host without one, which the Dockerfile handles.
RENDER_GID          ?=$(shell getent group render | cut -d: -f3)
DOCKER_BUILD_OPTS   ?=

.PHONY: build-image
build-image:
	docker build $(DOCKER_BUILD_OPTS) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		--build-arg ROS2_DISTRO=$(ROS2_DISTRO) \
		--build-arg APT_MIRROR=$(APT_MIRROR) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg RENDER_GID=$(RENDER_GID) .
