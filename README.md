# 本代码基于深势科技Uni-3DAR进行开发
```
@article{lu2025uni3dar,
  author    = {Shuqi Lu and Haowei Lin and Lin Yao and Zhifeng Gao and Xiaohong Ji and Weinan E and Linfeng Zhang and Guolin Ke},
  title     = {Uni-3DAR: Unified 3D Generation and Understanding via Autoregression on Compressed Spatial Tokens},
  journal   = {Arxiv},
  year      = {2025},
}
```
本模型为第三届世界科学智能大赛材料设计赛道优胜奖作品
本SAIS-Uni-3DAR模型，可基于量化性质数值（如分子能量等单一数值条件）条件生成小分子

## 训练过程使用的数据
Uni-3DAR本身的数据集中，每一个分子由一行字典构成，只需在其中加入一个键"properties"，并赋予相应值，即可启用该变种模型

## 训练过程的环境
4090D
处理器: 112核心
内存: 256G
GPU: NVIDIA 4090D GPU卡 * 4
GPU显存：24G
cuda版本：12.4

环境配置见requirements.txt

有些包无法通过pip install [名字] 进行直接安装，故提供我的环境配置操作步骤供参考
若想复现我的代码，建议严格按照我的配置顺序进行环境配置，以免发生未知错误（大佬随意）

我的镜像环境配置操作如下(在通过Dockerfile创建镜像成功后)
### 创建conda虚拟环境
```
conda create -n pytorch2.6_cuda118 python=3.12.0
```
### 进入虚拟环境，run.sh中加入了source activate pytorch2.6_cuda118来自动激活虚拟环境
```
conda activate pytorch2.6_cuda118
```
### 安装torch及cuda
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```
### 安装flash-attention 链接：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
### 安装Uni-Core
```
git clone https://github.com/dptech-corp/Uni-Core
cd /Uni-Core-main
pip install .
```
### 安装其他包
```
pip install numba pandas scikit-learn
```
进行以上操作后环境配置完毕


## 其他附加说明
在Uni-3DAR源代码的基础上，删除/注释掉了筛选生成分子以及其他不必要的功能，增加了基于量化化学性质的条件生成功能：

在/uni3dar/data中
新建molecule_energy_data_utils.py与molecule_energy_grid_dataset.py
修改了atom_data.py、atom_dictionary.py、grid_dataset.py
删除源代码中的crystal_data_utils.py、crystal_grid_dataset.py、protein_grid_dataset.py

在/uni3dar/models中
修改了uni3dar_sampler.py与uni3dar.py
删除了diffusion文件夹与diffusion_prediction_head.py

在/uni3dar/tasks中
修改了uni3dar.py

在/uni3dar中
修改了inference.py


鉴于修改代码量较大，在修改的代码中，增添的代码会有详细的中文注释，并且注释掉了源代码中不需要的代码，方便与源代码比对
(英文注释为源代码本身所有)

源代码链接：https://github.com/dptech-corp/Uni-3DAR



## 训练说明
submit.sh与ini_submit.sh均设置为初始训练参数，前者为slurm系统的提交脚本版本
## 复现训练命令
bash ini_submit.sh [训练集与验证集所在的文件夹路径] [自定义训练任务名称]
在/results/中可以看到以[自定义训练任务名称]命名的文件夹，保存的模型权重及其他信息均在其中

## 生成说明
sample.sh与ini_sample.sh均设置为初始训练参数，前者为slurm系统的提交脚本版本
复现训练命令
bash ini_sample.sh [选取的模型权重文件的路径]

注意生成脚本中merge_level、layer、emb_dim、head_num必须与训练脚本中的保持一致

## 传参说明
主要传参设置在/uni3dar/tasks/uni3dar.py与/uni3dar/models/uni3dar.py中

调参过程只涉及以下九个传参的变动，其余传参均不变
merge_level、layer、emb_dim、head_num、batch_size、batch_size_valid #这六个传参为训练脚本与生成脚本共有
tree_temperature、atom_temperature、xyz_temperature #这三个传参为生成脚本独有

训练脚本文件路径传参：
--user-dir ./uni3dar # 此处为相对路径，更改脚本路径的话，需要指定uni3dar文件夹路径
--train-subset train # 训练集名称，例如train.lmdb，则写train，如果是traintrain.lmdb，则写traintrain
--valid-subset valid # 验证集名称

生成脚本文件路径传参：
--user-dir ./uni3dar # 此处为相对路径，更改脚本路径的话，需要指定uni3dar文件夹路径
--train-subset train # 训练集名称，例如train.lmdb，则写train，如果是traintrain.lmdb，则写traintrain
--valid-subset valid # 验证集名称
--condition-file /saisdata/input_condition.csv # 条件集文件路径

另外生成submit.pkl的路径需要在/uni3dar/inference.py中的第323行处修改，如下所示
    with open("/saisresult/submit.pkl", "wb") as f:
        pickle.dump(parsed_molecules, f)
    print(parsed_molecules)
print函数用于直接屏幕输出所有生成分子的字典，可注释掉


因为本代码使用Uni-Core框架，所以沿用了Uni-Core框架中的传参
