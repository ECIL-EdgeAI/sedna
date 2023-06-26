FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-dev -y

COPY ./lib/requirements.txt /home
COPY ./lib/requirements.dev.txt /home

# install requirements of sedna lib
RUN pip install -r /home/requirements.txt
RUN pip install -r /home/requirements.dev.txt
RUN pip install torchvision~=0.13.0
RUN pip install Pillow
RUN pip install tqdm
RUN pip install protobuf~=3.20.1
RUN pip install matplotlib
RUN pip install python-multipart
RUN pip install tensorboard
RUN pip install watchdog
RUN pip install imbalanced-learn
RUN pip install scikit-image

RUN apt-get install wget -y
RUN wget -P /root/.cache/torch/hub/checkpoints/ --no-check-certificate https://download.pytorch.org/models/resnet18-5c106cde.pth

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY ./examples/lifelong_learning/cityscapes  /home/work/
WORKDIR /home/work/RFNet

ENTRYPOINT ["python"]