import  argparse
from transformers import AutoTokenizer
from  model import chat
from  model import VisualGLMModel
from sat.model.mixins import CachedAutoregressiveMixin

tokenizer = AutoTokenizer.from_pretrained("../visualglm-6b", trust_remote_code=True)
model, model_args = VisualGLMModel.from_pretrained("../visualglm-6b",args=argparse.Namespace(fp16=True, skip_init=True))

model.add_mixin('auto_regressive', CachedAutoregressiveMixin)
image_path='./fewshot-data/龙眼.jpeg'
promote='描述这张图片'
responses,history,cache_image= chat(image_path,model,tokenizer,promote,history=[])
print(responses)