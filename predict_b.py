import os
import torch
import json
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLProcessor
from bilibili_api import video, sync
from qwen_vl_utils import process_vision_info
import random  # 添加在文件开头的import部分

MODEL_NAME = "/data/home/models/Qwen2-VL-7B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bvid', type=str, required=True)  # 新增参数，用于接收单个视频ID
    args = parser.parse_args()
    return args

async def get_video_info(bvid):
    try:
        v = video.Video(bvid=bvid)
        info = await v.get_info()
        pages = await v.get_pages()
        cid = pages[0]['cid']
        video_url = await v.get_download_url(cid=cid)
        
        # 直接获取第一个可用的视频流
        video_stream = video_url['dash']['video'][0]

        return {
            'title': info['title'],
            'content': info['desc'],
            'video_url': video_stream['baseUrl'],
            'duration': info['duration']
        }
    except Exception as e:
        raise Exception(f"获取视频信息失败: {str(e)}")

def load_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor

def predict_video(video_info, model, processor):
    try:
        duration = video_info['duration']
        target_frames = 32
        fps = max(1, min(2, target_frames / duration)) / 4
        
        messages = [
            {
                "role": "system",
                "content": """你是一位专业的视频内容分析师。请观看视频并判断是否为标题党。请务必详细说明你的判断理由。

你必须包含以下内容：
1. 视频的主要内容是什么
2. 你看到了哪些具体画面
3. 这些内容与标题的关系
4. 最后给出明确的判断：是否是标题党"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_info['video_url'],
                        "max_pixels": 360 * 420,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": f"这个视频的标题是：「{video_info['title']}」，请分析这个视频是否是标题党，记得详细描述你看到的画面内容。"
                    }
                ]
            }
        ]

        # 准备推理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 生成回答
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,  # 增加生成的token数量
            temperature=0.7,      # 添加一些随机性
            do_sample=True       # 启用采样
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 直接打印结果
        print("\n" + "="*50)
        print(f"视频标题：{video_info['title']}")
        print(f"视频简介：{video_info['content'][:100]}...")  # 只显示前100个字符
        print("-"*50)
        print("AI分析结果：")
        print(response)
        print("="*50 + "\n")
        
        return response

    except Exception as e:
        print(f"处理视频时出错：{str(e)}")
        return f"处理失败：{str(e)}"
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    random.seed(args.seed)
    
    print("正在加载模型和processor...")
    model, processor = load_model_and_processor()
    
    try:
        print(f"\n正在处理视频ID: {args.bvid}")
        video_info = sync(get_video_info(args.bvid))
        result = predict_video(video_info, model, processor)
        
        # 保存结果
        final_result = {
            "bvid": args.bvid,
            "title": video_info['title'],
            "content": video_info['content'],
            "ai_analysis": result
        }
        
        with open('video_predict_result.json', 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"处理视频时出错：{str(e)}")
        
    finally:
        # 清理显存
        torch.cuda.empty_cache()
