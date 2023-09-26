FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt update
RUN apt-get update
RUN apt install nano -y
RUN apt install git -y
RUN apt install wget -y

RUN pip install tensorboard==2.12.2
RUN pip install scipy==1.10.1
RUN pip install numpy==1.22
RUN pip install git+https://github.com/VLL-HD/FrEIA.git
RUN pip install pyyaml
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install opencv-python


# from f1tenth_gym
ENV LIBGL_ALWAYS_INDIRECT=1
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update --fix-missing && \
    apt-get install -y \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1-mesa-dev \
                    mesa-utils \
                    libglu1-mesa-dev \
                    fontconfig \
                    libfreetype6-dev

RUN pip3 install --upgrade pip
RUN pip3 install PyOpenGL \
                 PyOpenGL_accelerate

COPY . /workspace

RUN cd /workspace/f1tenth_gym && \
    pip3 install -e .
RUN cd ..

RUN useradd -m -d /home/user -u 1004 user
USER root
RUN chown -R user:user /workspace
USER 1004:1004
ENV PATH="${PATH}:/home/user/.local/bin"

ENTRYPOINT ["/bin/bash"]
