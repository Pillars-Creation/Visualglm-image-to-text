# Visualglm-image-to-text
##介绍
使用了清华的Visualglm语言模型进行lora finetune，做了个简单到面相预测，补充了训练需要的一些文件，以及预测文件。

## 样例
对面相进行预测

<img width="523" alt="image" src="https://github.com/qjzcy/Visualglm-image-to-text/assets/19749009/96a07061-90ac-4d3d-8b0e-39976da7564a">

<img width="1303" alt="image" src="https://github.com/qjzcy/Visualglm-image-to-text/assets/19749009/634f2c7c-2209-4c70-9f84-b1dde3dda431">



## 使用

### 模型推理

使用pip安装依赖
```
pip install -r requirements.txt
```
尽量使用标准PyPI源以下载较新的sat包，TUNA源等可能同步较慢。`pip install -i https://pypi.org/simple -r requirements.txt`。
此时默认会安装`deepspeed`库（支持`sat`库训练），此库对于模型推理并非必要，同时部分Windows环境安装此库时会遇到问题。如果想绕过`deepspeed`安装，我们可以将命令改为
```
pip install -r requirements_wo_ds.txt
pip install --no-deps "SwissArmyTransformer>=0.3.6"
```

### 模型微调
1，需要在checkpints/300目录下下载mp_rank_00_model_states.pt文件，获取途径如下

wget https://huggingface.co/wangrongsheng/XrayGLM-300/resolve/main/300/mp_rank_00_model_states.pt

2，visualglm-6b 文件下载路径如下

[https://huggingface.co/THUDM/visualglm-6b/tree/main]

然后执行
bash  finetune_visualglm.sh

### 模型预测
原模型：python predict.py

fintune后模型：python predict_lora.py

##注意文件里路径都需要注意做相应调整，改到自己目录下。

# Visualglm-image-to-text