import os
import torch
import json
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from bilibili_api import video, sync
from qwen_vl_utils import process_vision_info
import random

MODEL_NAME = "/data/home/models/Qwen2-VL-7B-Instruct"

class VideoDataset(Dataset):
    def __init__(self, data_path, processor):
        self.processor = processor
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {
                "role": "system",
                "content": """你是一位专业的视频内容分析师。请观看视频并判断是否为标题党。请务必详细说明你的判断理由。"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": item['video_url'],
                        "max_pixels": 360 * 420,
                        "fps": 0.25,
                    },
                    {
                        "type": "text",
                        "text": f"这个视频的标题是：「{item['title']}」，请分析这个视频是否是标题党。"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": item['label']  # 人工标注的答案
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    return parser.parse_args()

def train():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("正在加载模型和processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    print("准备训练数据...")
    train_dataset = VideoDataset(args.train_data, processor)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("开始训练...")
    trainer.train()
    
    print("保存模型...")
    trainer.save_model()

if __name__ == "__main__":
    train()
