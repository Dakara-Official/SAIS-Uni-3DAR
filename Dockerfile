FROM continuumio/miniconda3

# 如有安装其他软件的需求
RUN apt-get -y update && apt-get -y install curl

# 复制代码到镜像仓库
COPY . /app

# 指定工作目录
WORKDIR /app

# 容器启动运行命令
CMD ["bash", "run.sh"]