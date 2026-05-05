---
title: "My First Post"
date: 2026-05-05T12:32:21+08:00
draft: false
categories: ["2d-vision"]
tags: ["YOLO", "CUDA", "PyTorch"]
---


# This is the first local blog, which in order to test the develop entirment


## 2.cuda安装（11.1）
首先应确定pytorch支持什么版本的cuda再安装，这里以11.1cuda为例  

去nv官网搜索cuda11.1 toolkit  选择Linux ->x86_64 ->ubuntu ->**20.04** ->runfile(local)

![alt 实例图片](https://img-blog.csdnimg.cn/4c20c19d829d4df0936abcb696763a10.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6K6h566X5py66KeG6KeJ5LuO6Zu25a2m,size_20,color_FFFFFF,t_70,g_se,x_16)

copy下面出现的命令  
一般是wget.......  
sudo sh cuda.....  
自行安装 会比较慢  

sh过程中会弹出cuda的安装器，直接continue ->accept 最后取消勾选driver（第一个） 因为前面我们已经装过了，下面四个勾上，如果让你选择yes or no全选yes，install按回车
![alt 图片示例](https://img-blog.csdnimg.cn/a165fb70921749a6933b508694d81170.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6K6h566X5py66KeG6KeJ5LuO6Zu25a2m,size_12,color_FFFFFF,t_70,g_se,x_16)

等待.......

### 配置环境变量
进入根目录 
```bash
    cd ~  
    nano .bashrc  #用nano或是gedit打开bashrc文件，尽量不去碰vim  
 ```
在文末添加  
```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CPUTI/lib64  
    export CUDA_HOME=/usr/local/cuda/bin  
    export PATH=$PATH:$LD_LIBRARY_PATH:$CUDA_HOME  
```
保存（ctrl+o）  
```bash
    echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc  
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc  
    source ~/.bashrc #刷新  
```
退出(ctrl+x)

### 检查是否安装完成
nvcc -V（大写）  
如果正确输出五行并且打印版本正确就行了  
可以看看你安装的四个软件是否在软件目录中  

![alt 图片示例](https://img-blog.csdnimg.cn/9c49e83ff14e494ea1043fb5499406ed.png)
## 3.pytorch安装(1.8.1 cuda加速版本)

确认已经安装正确的cuda全家桶 ，驱动也安装完毕  
进入pytorch官网搜索pytorch1.8.1  
一定要使用pip安装方式！！！ 不要瞎搞conda  


更新pip 
```bash
    pip install --upgrade pip
```
找到cuda11.1版本的pip安装方式，在终端键入命令  
```bash
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

等待安装完成  

终端键入  
```bash
    python3 -c "import torch; print(torch.cuda.is_available())"  
```
  
若输出True，则安装完毕  
若False，重装  

# ultralytics相关的准备

主要会介绍yolov5和yolov8两种模型的本地部署和训练，此篇内容较简单，具体操作请查阅官方文档https://docs.ultralytics.com/zh  

## yolov5的本地部署和训练

确保在你的根目录下  
```bash
    git clone https://github.com/ultralytics/yolov5  # clone repository  
    cd yolov5  
    pip install -r requirements.txt  # install dependencies  
直接在github上克隆下载整个文件夹
```
### 训练

命令：
```bash
    python3 train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128  
```
**该命令仅供参考**  
epochs：训练轮数  
weights：权重  
yolov5n：所使用的神经网络模型类型  
batch-size：一批量的照片数量（请根据显卡显存以及所选神经网络模型类型决定，宁小勿大）  
除此之外，你还需指定数据集的路径等等，此篇概不赘述，官方文档有详细解答  

### 用detect进行推理

命令：
```bash
    python3 detect.py --weights yolov5s.pt --source img.jpg  
```
脚本 detect.py 用于对各种来源进行多功能推理。它能自动获取 模型 从最新的YOLOv5 释放 并轻松保存结果。  


**warning:本文只是概述了yolov5模型的使用，但绝不代表其功能仅限于此，请务必查阅官方文档了解更多**  

## yolov8和ultralytics软件包
本地部署：
```bash
    pip install ultralytics
```
![alt 图片示例](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

具体功能查阅  
https://github.com/ultralytics/ultralytics  
https://github.com/ultralytics/assets/releases/tag/v8.2.0  
与yolov5有所不同

