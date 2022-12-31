# 使用sedna启动云边AI联合推理

PS：执行下面操作时你需要在sedna根目录

## 构建arm64镜像

- little镜像： docker build -f examples/joint-inference-helmet-detection-little-arm64.Dockerfile . -t poorunga/sedna-example-joint-inference-helmet-detection-little-arm64:v0.3.0

- big镜像： docker build -f examples/joint-inference-helmet-detection-big-arm64.Dockerfile . -t poorunga/sedna-example-joint-inference-helmet-detection-big-arm64:v0.3.0

## 按照指南：https://github.com/kubeedge/sedna/tree/main/examples/joint_inference/helmet_detection_inference 把大小模型下了，目录创建了

## 创建联合推理任务

```shell
CLOUD_NODE="cloud-node-name"
EDGE_NODE="edge-node-name"

kubectl apply -f examples/joint_inference/helmet_detection_inference/helmet_detection_all_arm64.yaml
```

## EasyDarwin 需要用arm64版本的，已经放进来了，在sedna根目录下

---

# 使用docker启动云边AI联合推理

思路：
1. 主要就是在边侧pod内使用wondershaper去限制边侧pod图片的upload速率
2. 由于sedna启动的边侧pod强行是主机网络的，如果在主机网络的pod内使用wondershaper限制eth0，会影响宿主机的网络
3. 所以必须用docker run的方式直接启动边侧pod，避免它是主机网络启动
4. 由于docker容器技术具备网络隔离能力，在此时边侧pod内使用wondershaper限制eth0，就只作用在pod自身的eth0网卡，并不会影响外层宿主机，即可达到我们的目的
5. 流程总结：docker启动云上pod -> docker启动边侧pod（非主机网络） -> exec进入边侧pod -> wondershaper限制eth0 -> python启动AI代码

## 启动云侧pod

```shell
docker run --net=host -d \
-e input_shape=544,544 \
-e DATA_PATH_PREFIX=/ \
-e MODEL_URL=/data/big-model/yolov3_darknet.pb \
-e BIG_MODEL_BIND_PORT=5000 \
-v /data:/data \
poorunga/sedna-example-joint-inference-helmet-detection-big-arm64:v0.3.1
```

## 启动带wondershaper的边侧pod，并且开启限速

1.由于边侧pod不再是主机网络，所以视频流必须启动在docker0网桥172.17.0.1上，那样边侧pod才能访问得到：
```shell
ffmpeg -re -stream_loop -1 -i /data/video/video.mp4 -vcodec libx264 -f rtsp rtsp://172.17.0.1/video1
```

2.启动边侧pod：
```shell
docker run -it \
--privileged \
--entrypoint /bin/bash \
-e input_shape=416,736 \
-e video_url=rtsp://172.17.0.1/video1 \
-e all_examples_inference_output=/data/output/output \
-e hard_example_cloud_inference_output=/data/output/hard_example_cloud_inference_output \
-e hard_example_edge_inference_output=/data/output/hard_example_edge_inference_output \
-e LC_SERVER=http://172.17.0.1:9100 \
-e SUPER_BIG_MODEL_IP=202.170.91.188 \
-e BIG_MODEL_IP=202.170.91.188 \
-e BIG_MODEL_PORT=5000 \
-e INFERENCE_TIMEOUT=3699 \
-e HEM_NAME=IBT \
-e HEM_PARAMETERS='[{"key":"threshold_img","value":"0.9"},{"key":"threshold_box","value":"0.9"}]' \
-e DATA_PATH_PREFIX=/ \
-e MODEL_URL=/data/little-model/yolov3_resnet18.pb \
-v /data:/data \
-v /joint_inference/output:/data/output \
poorunga/sedna-example-joint-inference-helmet-detection-little-arm64:v0.3.1-wondershaper
```

3.(此时会进入到边侧pod容器环境)使用wondershaper进行限速
```shell
cd /home/wondershaper
./wondershaper -a eth0 -u 64

cd /home/work
python3 infer.py
```
