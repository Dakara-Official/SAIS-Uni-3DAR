# 本代码基于深势科技Uni-3DAR进行开发
```
@article{lu2025uni3dar,
  author    = {Shuqi Lu and Haowei Lin and Lin Yao and Zhifeng Gao and Xiaohong Ji and Weinan E and Linfeng Zhang and Guolin Ke},
  title     = {Uni-3DAR: Unified 3D Generation and Understanding via Autoregression on Compressed Spatial Tokens},
  journal   = {Arxiv},
  year      = {2025},
}
```

0. 数据预处理+训练说明+生成说明+传参说明
①数据预处理
以下文件均在/app/data_preprocess/路径下，使用以下代码时，需要进入代码内部自行更改输入文件路径与输出文件路径

原始数据为sais赛方提供的复赛数据集competition_round2.pkl

使用pkl2npz.py代码，按照8:2的比例，将competition_round2.pkl转换成train.npz与valid.npz

使用npz2txt.py代码，将train.npz、valid.npz转换成train_npz.txt、valid_npz.txt，每一行有一个字典，对应一个分子结构 #数据预处理的中间文件使用txt格式，方便观察数据

使用npz2lmdb.py代码，对原字典进行转换，将train_npz.txt、valid_npz.txt转换成train_lmdb.txt、valid_lmdb.txt #将原子序数(charges)转换成元素符号(atom_type)，将二维列表坐标(positions)转换成三维列表坐标(atom_pos)，将量化化学性质(properties)正常传递，不传递原子数量(natoms)

最后使用txt2lmdb.py代码，将train_lmdb.txt转换train.lmdb，将valid_lmdb.txt转换valid.lmdb，作为训练模型的输入文件 #会附带生成train.lmdb-lock、valid.lmdb-lock，不用管这俩文件

针对数据集中全部的分子，提取其中properties最大值与最小值，作为参数写进训练与生成脚本，作为properties归一化的计算依据，对读取数据中的properties进行基本的min-max归一化。

②训练说明
/app/training_code/中的submit.sh与ini_submit.sh均设置为初始训练参数，前者为slurm系统的提交脚本版本
复现训练命令
bash ini_submit.sh [训练集与验证集所在的文件夹路径] [自定义训练任务名称]
在/app/training_code/results/中可以看到以[自定义训练任务名称]命名的文件夹，保存的模型权重及其他信息均在其中

③生成说明
/app/training_code/中的sample.sh与ini_sample.sh均设置为初始训练参数，前者为slurm系统的提交脚本版本
复现训练命令
bash ini_sample.sh [选取的模型权重文件的路径] # 生成脚本/app/run.sh为了满足比赛条件，已经将路径嵌入脚本内
生成的分子保存在/saisresult/submit.pkl中

注意生成脚本中merge_level、layer、emb_dim、head_num必须与训练脚本中的保持一致

④传参说明
主要传参设置在/app/training_code/uni3dar/tasks/uni3dar.py与/app/training_code/uni3dar/models/uni3dar.py中

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

另外生成submit.pkl的路径需要在/app/training_code/uni3dar/inference.py中的第323行处修改，如下所示
    with open("/saisresult/submit.pkl", "wb") as f:
        pickle.dump(parsed_molecules, f)
    print(parsed_molecules)
print函数用于直接屏幕输出所有生成分子的字典，可注释掉


因为本代码使用Uni-Core框架，所以沿用了Uni-Core框架中的传参


1. 训练过程使用的数据
有且仅有sais赛方提供的复赛数据集competition_round2.pkl


2. 训练过程的环境
4090D
处理器: 112核心
内存: 256G
GPU: NVIDIA 4090D GPU卡 * 4
GPU显存：24G
cuda版本：12.4

环境中配置的所有包及对应版本见/app/requirements/requirements.txt


有些包无法通过pip install [名字] 进行直接安装，故提供我的环境配置操作步骤供参考
若想复现我的代码，建议严格按照我的配置顺序进行环境配置，以免发生未知错误

我的镜像环境配置操作如下(在通过Dockerfile创建镜像成功后)

conda create -n pytorch2.6_cuda118 python=3.12.0 #创建conda虚拟环境

conda activate pytorch2.6_cuda118 #进入虚拟环境，run.sh中加入了source activate pytorch2.6_cuda118来自动激活虚拟环境

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118  #安装torch及cuda

cd /app/requirements #进入配置文件夹

pip install flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl  #安装flash-attention 链接：https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

cd /Uni-Core-main  #进入Uni-Core文件夹 链接：https://github.com/dptech-corp/Uni-Core

pip install .   #安装Uni-Core

pip install numba pandas scikit-learn #安装其他包

进行以上操作后环境配置完毕


3. 其他附加说明
在Uni-3DAR源代码的基础上，删除/注释掉了筛选生成分子以及其他不必要的功能，增加了基于量化化学性质的条件生成功能：

在/app/training_code/uni3dar/data中
新建molecule_energy_data_utils.py与molecule_energy_grid_dataset.py
修改了atom_data.py、atom_dictionary.py、grid_dataset.py
删除源代码中的crystal_data_utils.py、crystal_grid_dataset.py、protein_grid_dataset.py

在/app/training_code/uni3dar/models中
修改了uni3dar_sampler.py与uni3dar.py
删除了diffusion文件夹与diffusion_prediction_head.py

在/app/training_code/uni3dar/tasks中
修改了uni3dar.py

在/app/training_code/uni3dar中
修改了inference.py


鉴于修改代码量较大，在修改的代码中，增添的代码会有详细的中文注释，并且注释掉了源代码中不需要的代码，方便与源代码比对
(英文注释为源代码本身所有)

源代码链接：https://github.com/dptech-corp/Uni-3DAR/tree/bba82f5ffdcd85963e1dff68a52415cbf046e6c7


4. 建模策略
Uni-3DAR的核心创新是将3D结构通过层次化token化方法统一表示为一维token序列。
具体步骤：
​​①层次化八叉树压缩​​：利用八叉树递归细分3D空间，非空区域生成token，保留空间位置信息。
​②​精细结构token化​​：在最后层非空区域引入3D patch概念，将局部结构离散化为token(类似图像patch)。
​​③二级子树压缩​​：将父节点与8个子节点合并为一个token(256种状态)，减少token数量8倍。

​​因token序列动态展开，传统自回归位置推断失效。创新性引入Masked Next-Token Prediction策略：复制每个token并掩码其中一个，使模型能利用位置信息预测下一个token。
生成任务(掩码token预测)、理解任务(原子级属性预测、分子级分类)通过特殊token(如[EoS])在单一模型内协同训练，实现多任务统一。


5. 描述符方案
输入描述符
​​原子类型/位置​​：通过嵌入层编码原子类型和三维坐标(分xyz三个方向嵌入)。
​​层次信息​​：八叉树层级嵌入（cur_level）。
​​树/空间索引​​：多树/空间结构嵌入（tree_emb, space_emb）。
​​计数信息​​：剩余原子/令牌数嵌入(remaining_atom_emb, remaining_token_emb)。
位置编码：3D RoPE
​​旋转位置编码扩展​​：将RoPE推广到3D空间，每轴独立计算频率，保留空间相对位置。
​​计算优化​​：通过八叉树层级和坐标计算频率，缓存结果加速推理。


6. 调参过程
初始训练参数如/app/training_code/submit.sh中所示
初始生成参数如/app/training_code/sample.sh中所示
基于以上初始参数进行调整

①调整了生成参数，tree_temperature从0.9改成0.4，排行榜分数从17.0782上升至44.9914。

②调整训练参数，emb_dim从768改成384，head_num从12改成6，在batch_size=16与batch_size=64的情况下，分别训练两个模型，前者排行榜分数19.4148，后者排行榜分数1.8266。分数下降显著，故决定不降低emb_dim与head_num，并保持batch_size=16。

③调整训练参数，layer从12改成18，emb_dim=768，head_num=12，排行榜分数提高至45.0433。

④调整生成参数，使用分数44.9914对应的训练模型，将tree_temperature从0.4改成0.15，排行榜分数上升至45.7714

⑤调整训练参数，layer=12，emb_dim从768改成1024，head_num从12改成16，排行榜分数提高至47.8430

⑥调整训练参数，layer从12改成24, emb_dim=1024, head_num=16，排行榜分数提高至51.8863

⑦调整训练参数，merge_level从6改成8，分数微微下降，故决定保持merge_level=6

⑧调整生成参数，tree_temperature从0.15改成0.10，排行榜分数提高至52.4278

⑨调整生成参数，atom_temperature与atom_temperature从0.3改成0.6，排行榜分数提高至52.4306


7. 超参数优化的过程
无


8. 外部数据使用情况
无