# ResNet50-paddlepadlle2.2

> 目前用于遮挡、丢帧、正常视频分类模型训练 

## 安装训练环境

- 安装cuda

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
- 安装cudnn

```bash
wget 'https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.0_20201106/cudnn-11.0-linux-x64-v8.0.5.39.tgz?5I9X8UVBZqDfDbF85gvfQV3rMNlYn0DtWgiPkkubl-CgDhtyN7qAU1hm_UetZLaJRHZzo4AoSfeok0TMgEwauz5dJQMjX_lWJxFeWAB66noghDPgTcSTuDD40NzkBYLXQJ9_tYdfOEib8eeAtejR1ErR-qHmJlfyOqV8Lm3E2c1lRpe1VPd0Jqo4wuCr46DdHg3zGXccBrr3kug&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczpcL1wvd3d3Lmdvb2dsZS5jb21cLyJ9' -O cudnn-11.0-linux-x64-v8.0.5.39.tgz
```

- 安装paddle

```bash
python -m pip install paddlepaddle-gpu==2.2.2.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

#check install
import paddle
paddle.utils.run_check()
#log
pip install --upgrade --pre  visualdl
```

## 下载预训练模型参数

```bash
wget 'https://bj.bcebos.com/v1/ai-studio-online/5850d297544d4c31b1d86973a1299c1f4498c1de314742a4a08bff2b4a8c40b6?responseContentDisposition=attachment%3B%20filename%3DResNet50_pretrained.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2019-04-29T05%3A12%3A58Z%2F-1%2F%2F93e7c9cf8f749a1f7767526a0cdaa736a89729961a1fdd00f0c6918cccf089bf' -O ResNet50_pretrained.zip
unzip ResNet50_pretrained.zip -d .
```

## 分割数据集

```bash
python dataSplit.py
```

## 开始训练

```bash
python main.py
```
