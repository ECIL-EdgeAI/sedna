
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
