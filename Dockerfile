FROM ubuntu:16.04

MAINTAINER Jian Wu <hellojianwu@gmail.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    python -m ipykernel.kernelspec

# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp27-none-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# TensorBoard
EXPOSE 6006

COPY dist-dqn-trainer.py /
COPY start-dqn-training.sh /

WORKDIR /

CMD ./start-dqn-training.sh --ps_hosts=${PS_HOSTS} --worker_hosts=${WORKER_HOSTS} --job_name=${JOB_NAME} --task_index=${TASK_INDEX} --sync_flag=${SYNC_FLAG}
