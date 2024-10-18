import asyncio
import json
import argparse
import dill
from vertexai.generative_models import GenerativeModel, Part
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from utils import load_json, collect, load_hotpotqa, json_save, load_dill, load_hotpotqa, load_jsonl, load_collie
import random
from datasets import load_dataset
import time
import os

base_url="http://60.204.212.177:3000/v1"
api_key="your_api_key"
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

# 设置并发限制
MAX_RETRIES = 10
BASE_DELAY = 1
MAX_DELAY = 60
MAX_CONCURRENT = 64


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=60))
async def get_chat_completion(message: str, semaphore, retry_count=0) -> str:
    try:
        async with semaphore:  # 使用传入的信号量限制并发
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                timeout=80
            )
            response_result = response.choices[0].message.content

            return {'response_result': response_result, 'message': message}
            # return response_result
    except Exception as e:
        print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
        raise


async def request_model(prompts):

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async def wrapped_get_chat_completion(prompt):
        try:
            return await get_chat_completion(prompt, semaphore)
        except Exception as e:
            print(f"Task failed after all retries with error: {e}")
            return None

    tasks = [wrapped_get_chat_completion(prompt) for prompt in prompts]
    
    results = []
    for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
        result = await future
        results.append(result)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for o1 capability analysys")

    parser.add_argument("--dataset_name" , type = str , default = 'hotpotqa')
    parser.add_argument("--model_name" , type = str , default = 'GPT4o')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name


    # dataset_name = 'hotpotqa'
    # dataset_name = 'collie'
    # model_name = 'GPT4o'
    # model_name = 'Claude'
    # model = "text-embedding-3-large"
    global model
    if model_name == 'GPT4o': 
        model = "gpt-4o-2024-08-06"
    elif model_name == 'Claude':
        model = 'claude-3-5-sonnet-20240620'
    elif model_name == 'o1':
        model = 'o1-preview'
    elif model_name == 'o1_mini':
        model = 'o1-mini'
    
    
    if dataset_name == 'hotpotqa':
        all_data = load_json('./data/hotpotqa_sentence_bert_filter.json')#[0:10]
    elif dataset_name == 'collie':
        all_data = load_dill('./data/collie_sentence_bert_filter.dill')#[10:11]
    elif dataset_name == 'aime':
        data_aimo = load_dataset('AI-MO/aimo-validation-aime')['train']#.select(range(7))
    elif dataset_name == 'usaco_bronze':
        all_data = load_jsonl('./data/usaco_bronze.jsonl')

    prompts = []
    if dataset_name == 'hotpotqa':
        prompt2item = {}
        for item in all_data:
            question = item['question']
            context = item['context']
            answer = item['answer']
            content = f"Give you a question: {question}, and a context: {context}, please answer the question using the content within the context."
            prompts.append(content)
            prompt2item[content] = item
        responses = asyncio.run(request_model(prompts))
        
    elif dataset_name == 'collie':
        prompt2item = {}
        for item in all_data:
            content = item['prompt']
            prompts.append(content)
            prompt2item[content] = item
        responses = asyncio.run(request_model(prompts))
    elif dataset_name == 'aime':
        all_data = []
        
        prompt2item = {}
        for item in data_aimo:
            problem = item['problem']
            prompts.append(problem)
            all_data.append(item)
            prompt2item[problem] = item

        responses = asyncio.run(request_model(prompts))
    elif 'usaco' in dataset_name:
        prompt2item = {}
        for item in tqdm(all_data):
            messages = item['messages']
            content = messages[0]['content']

            prompts.append(content)
            prompt2item[content] = item

        responses = asyncio.run(request_model(prompts))
    
    
    #根据下标检索item
    results = []
    for response in responses:
        response_result = response['response_result']
        prompt = response['message']
        item = prompt2item[prompt]
        item['response'] = response_result
        results.append(item)

    os.makedirs(f'./results/', exist_ok=True)

    if dataset_name == 'collie':
        with open(f'./results/{model_name}_{dataset_name}.dill', 'wb') as f:
            dill.dump(results, f)
    else:
        json_save(results, f'./results/{model_name}_{dataset_name}.json')
