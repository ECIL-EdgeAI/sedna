FROM ubuntu:18.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip libgl1-mesa-glx python3-h5py

RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install pandas

COPY ./lib/requirements.txt /home
RUN pip3 install -r /home/requirements.txt
RUN pip3 install opencv-python
RUN pip3 install Pillow
WORKDIR /root
COPY ./tensorflow-1.15.5-cp36-cp36m-linux_aarch64.whl /root
RUN pip3 install tensorflow-1.15.5-cp36-cp36m-linux_aarch64.whl
RUN pip3 install pycocotools

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python3"]

COPY examples/joint_inference/helmet_detection_inference/big_model/big_model.py  /home/work/infer.py
COPY examples/joint_inference/helmet_detection_inference/big_model/interface.py  /home/work/interface.py

CMD ["infer.py"]
