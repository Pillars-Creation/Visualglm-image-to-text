# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

from model import VisualGLMModel, chat
from finetune_visualglm import FineTuneVisualGLMModel
from sat.model import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="checkpoints/finetune-visualglm-6b-06-06-00-05/",
                        help='pretrained ckpt')
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True,
            device='cuda'
        ))
    model = model.to(torch.float32)
    model = model.eval()

    if args.quant:
        quantize(model.transformer, args.quant)

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    tokenizer = AutoTokenizer.from_pretrained("../chatglm-6b", trust_remote_code=True)
    with torch.no_grad():
        history = None
        cache_image = None
        image_path = './fewshot-data/虎眼.jpeg'
        query = '从鼻子描述这张图片人物可能的面相？'
        response, history, cache_image = chat(
            image_path,
            model,
            tokenizer,
            query,
            history=history,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            english=args.english,
            invalid_slices=[slice(63823, 130000)] if args.english else []
        )
        sep = 'A:' if args.english else '答：'
        print(query + ': ' + response.split(sep)[-1].strip())
        image_path = None

        query = '从眼睛看这张照片可能的面相？'
        response, history, cache_image = chat(
            image_path,
            model,
            tokenizer,
            query,
            history=history,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            english=args.english,
            invalid_slices=[slice(63823, 130000)] if args.english else []
        )
        sep = 'A:' if args.english else '答：'
        print(query + ': ' + response.split(sep)[-1].strip())

        query = '从嘴巴描述这张图片人物可能的面相？'
        response, history, cache_image = chat(
            image_path,
            model,
            tokenizer,
            query,
            history=history,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            english=args.english,
            invalid_slices=[slice(63823, 130000)] if args.english else []
        )
        sep = 'A:' if args.english else '答：'
        print(query + ': ' + response.split(sep)[-1].strip())

        query = '从天庭描述这张图片人物可能的面相？'
        response, history, cache_image = chat(
            image_path,
            model,
            tokenizer,
            query,
            history=history,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            english=args.english,
            invalid_slices=[slice(63823, 130000)] if args.english else []
        )
        sep = 'A:' if args.english else '答：'
        print(query + ': ' + response.split(sep)[-1].strip())


if __name__ == "__main__":
    main()

# python predict_lora.py