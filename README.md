# Visualglm-image-to-text
##介绍
使用了清华的Visualglm语言模型进行lora finetune，补充了预测文件，和训练需要的一些文件。

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

如果使用Huggingface transformers库调用模型（也需要安装上述依赖包！），可以通过如下代码（其中图像路径为本地路径）：
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
image_path = "your image path"
response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
print(response)
response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
print(response)
```

如果使用SwissArmyTransformer库调用模型，方法类似，可以使用环境变量`SAT_HOME`决定模型下载位置。在本仓库目录下：
```python
>>> import argparse
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> from model import chat, VisualGLMModel
>>> model, model_args = VisualGLMModel.from_pretrained('visualglm-6b', args=argparse.Namespace(fp16=True, skip_init=True))
>>> from sat.model.mixins import CachedAutoregressiveMixin
>>> model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
>>> image_path = "your image path or URL"
>>> response, history, cache_image = chat(image_path, model, tokenizer, "描述这张图片。", history=[])
>>> print(response)
>>> response, history, cache_image = chat(None, model, tokenizer, "这张图片可能是在什么场所拍摄的？", history=history, image=cache_image)
>>> print(response)
```
使用`sat`库也可以轻松进行进行参数高效微调。<!-- TODO 具体代码 -->

请注意，`Huggingface`模型的实现位于[Huggingface的仓库](https://huggingface.co/THUDM/visualglm-6b)中，`sat`模型的实现包含于本仓库中。

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