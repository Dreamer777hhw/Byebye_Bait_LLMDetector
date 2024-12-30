import os
import numpy as np
import torch
import re
import json
import argparse
import random
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoModelForCausalLM

MODEL_NAME = "/data/home/models/Qwen2-VL-7B-Instruct"

prompt = """你是一位专业的新闻审核员。你的任务是分析新闻是否为标题党。

标题党的特征包括：
1. 标题夸大其词，与实际内容严重不符
2. 使用情绪化、煽动性的词语
3. 故意制造悬念或误导读者
4. 断章取义，片面解读
5. 图文不符，使用夸张或误导性图片

请仔细分析新闻的标题、内容和图片，给出明确的判断（是/否）并详细说明理由。

你的回答必须按以下格式输出：

最终判断：[是/否]标题党

判断理由：
1. 标题分析：[分析标题是否夸大、情绪化、制造悬念]
2. 内容对比：[分析标题与内容的对应关系]
3. 图片分析：[分析图片与内容的关联性]
4. 整体评估：[总结性说明]"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'text_data' in item:
                        data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def evaluate_analysis(model, tokenizer, news_info, ai_analysis):
    eval_prompt = """你是一位资深的AI输出质量评估专家。请评估下面这段AI对新闻的分析是否准确、完整和可靠。

评估要点：
1. 是否给出了明确的"是/否"判断结论
2. 分析是否完整覆盖了所有要求的评估维度（标题夸大、情绪煽动、制造悬念、断章取义、图文关系）
3. 论据是否充分，是否有具体的文本和图片分析支持
4. 结论是否明确且合理

请按以下格式给出评估：

评分：[0-100分]

分析优点：
[列出分析中的优点]

存在问题：
[列出分析中的问题]

改进建议：
[具体的改进建议]

新闻信息：
{news_info}

AI分析内容：
{ai_analysis}
"""
    
    messages = [
        {
            "role": "system",
            "content": "你是一位专业的AI输出质量评估专家。"
        },
        {
            "role": "user",
            "content": eval_prompt.format(
                news_info=f"标题：{news_info['title']}\n内容：{news_info['content']}",
                ai_analysis=ai_analysis
            )
        }
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
    
    output = model.generate(
        input_ids=model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
    ]
    
    evaluation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return evaluation

def extract_score(evaluation_text):
    """从评估文本中提取分数"""
    score_pattern = r'评分[：:]\s*(\d+)'
    match = re.search(score_pattern, evaluation_text)
    if match:
        return int(match.group(1))
    return 0

def is_reliable(score, threshold=80):
    """判断分析是否可靠"""
    return score >= threshold

def generate_improved_analysis(model, tokenizer, news_info, previous_analysis, evaluation, analysis_history):
    improve_prompt = """你是一位专业的新闻审核员。请根据以下评估意见和历史分析记录，改进你的新闻分析。

请严格按照以下格式输出：

最终判断：[是/否]标题党

判断理由：
1. 标题分析：[分析标题是否夸大、情绪化、制造悬念]
2. 内容对比：[分析标题与内容的对应关系]
3. 图片分析：[分析图片与内容的关联性]
4. 整体评估：[总结性说明]

原始分析：
{previous_analysis}

评估意见：
{evaluation}

历史分析记录：
{history_summary}

新闻信息：
{news_info}

注意：
1. 必须给出明确的是/否判断
2. 分析要更加具体和深入
3. 避免模糊的表述
4. 确保结论有充分的支持证据"""
    
    messages = [
        {
            "role": "system",
            "content": "你是一位专业的新闻审核员。"
        },
        {
            "role": "user",
            "content": improve_prompt.format(
                previous_analysis=previous_analysis,
                evaluation=evaluation,
                history_summary=analysis_history,
                news_info=f"标题：{news_info['title']}\n内容：{news_info['content']}"
            )
        }
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
    
    output = model.generate(
        input_ids=model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def analyze_news_with_iteration(model, tokenizer, news_info, messages, max_iterations=3, score_threshold=80):
    """迭代分析新闻直到达到评分阈值或达到最大迭代次数"""
    iteration = 0
    best_score = 0
    best_analysis = None
    best_evaluation = None
    
    while iteration < max_iterations:
        # 生成分析
        if iteration == 0:
            # 首次分析
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(text)
            model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
            output = model.generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
            ]
            analysis = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # 基于评估改进分析
            analysis = generate_improved_analysis(model, tokenizer, news_info, best_analysis, best_evaluation, best_analysis)
        
        # 评估分析
        evaluation = evaluate_analysis(model, tokenizer, news_info, analysis)
        score = extract_score(evaluation)
        
        # 更新最佳结果
        if score > best_score:
            best_score = score
            best_analysis = analysis
            best_evaluation = evaluation
        
        # 检查是否达到阈值
        if score >= score_threshold:
            break
            
        iteration += 1
        print(f"迭代 {iteration}: 当前评分 {score}")
    
    return {
        "analysis": best_analysis,
        "evaluation": best_evaluation,
        "score": best_score,
        "iterations": iteration + 1,
        "is_reliable": best_score >= score_threshold
    }

def predict_news():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    news_data = read_jsonl('news_detection_data.jsonl')
    results = []
    
    for item in news_data:
        title = item['text_data']['title']
        content = item['text_data']['content']
        
        # 获取图片信息
        image_data = item.get('image_data', {})
        image_urls = image_data.get('image_urls', [])
        image_count = image_data.get('image_count', 0)
        
        # 构建多模态消息
        messages = [
            {
                "role": "system",
                "content": """你是一位专业的新闻审核员。你的任务是分析新闻是否为标题党。我会给你新闻的标题、正文内容和相关图片。

请你特别注意：
1. 标题与实际内容是否严重不符
2. 是否使用情绪化、煽动性的词语
3. 是否故意制造悬念或误导读者
4. 是否存在断章取义、片面解读
5. 新闻配图是否与内容相符，是否使用了夸张或误导性的图片

在分析时，请：
- 仔细观察每张配图的内容
- 判断图片与文字内容的关联度
- 评估图片是否客观真实地反映了新闻内容
- 考虑图文整体是否存在夸大或误导

请给出明确的判断（是/否）并详细说明理由，一定要包含对图片的具体分析。"""
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 处理文本和图片的混合内容
        content_parts = content.split('[IMAGE_')
        
        # 添加第一段文本(如果有)
        if content_parts[0].strip():
            messages[1]["content"].append({
                "type": "text",
                "text": f"新闻标题：{title}\n\n{content_parts[0].strip()}"
            })
        
        # 处理剩余的图文内容
        for i in range(1, len(content_parts)):
            part = content_parts[i]
            img_idx = int(part[0])
            text_content = part[2:].strip()
            
            # 添加图片和说明
            if img_idx < len(image_urls):
                messages[1]["content"].append({
                    "type": "image",
                    "image": image_urls[img_idx]
                })
                messages[1]["content"].append({
                    "type": "text",
                    "text": f"[图片{img_idx + 1}] 请仔细观察这张图片。"
                })
            
            # 添加后续文本
            if text_content:
                messages[1]["content"].append({
                    "type": "text",
                    "text": text_content
                })
        
        # 在最后添加分析请求
        messages[1]["content"].append({
            "type": "text",
            "text": "\n请结合以上所有图文内容，分析这篇新闻是否为标题党。请特别说明每张图片与新闻内容的关系，以及是否存在图文不符或误导性使用图片的情况。"
        })

        news_info = {
            "title": title,
            "content": content
        }
        
        # 使用迭代分析替代原来的单次分析
        result = analyze_news_with_iteration(
            model, 
            tokenizer, 
            news_info, 
            messages,
            max_iterations=3,  # 最大迭代次数
            score_threshold=80 # 评分阈值
        )
        
        # 构建完整的结果
        full_result = {
            "title": title,
            "content": content,
            "image_data": {
                "image_count": image_count,
                "image_urls": image_urls
            },
            "ai_analysis": result["analysis"],
            "evaluation": result["evaluation"],
            "score": result["score"],
            "iterations": result["iterations"],
            "is_reliable": result["is_reliable"]
        }
        results.append(full_result)
        
        print("\n新闻标题:", title)
        print("图片数量:", image_count)
        print("迭代次数:", result["iterations"])
        print("最终评分:", result["score"])
        print("AI判断:", result["analysis"])
        print("\nAI评估:", result["evaluation"])
        print("-" * 50)
    
    with open('news_predict_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    predict_news()
