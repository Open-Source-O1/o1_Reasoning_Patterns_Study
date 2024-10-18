import asyncio
import json
import argparse
import dill
from vertexai.generative_models import GenerativeModel, Part
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import load_json, collect, load_hotpotqa, json_save, load_dill, load_hotpotqa, load_jsonl
from datasets import load_dataset
import os


base_url="http://60.204.212.177:3000/v1"
api_key="your_api_key"
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

# 设置并发限制
MAX_RETRIES = 10
BASE_DELAY = 1
MAX_DELAY = 60
MAX_CONCURRENT = 64

def load_response_json(response):
    results = response
    results = results.replace('json', '')
    results = results.replace('\n', '')
    results = results.replace("```", '')
    
    step_data = json.loads(results)
    return step_data

def choose_best_response(messages, responses):
    if len(responses) == 1:
        return responses[0]
    else:
        # 获取整个prompt
        device = "cuda:0"
        prompts = []
        for item in messages:
            prompts.append(item['content'])
        prompt = ' '.join(prompts)

        convs = []
        for r in responses:
            convs.append(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": r}
            ]
            )
        
        scores = []
        for c in convs:
            with torch.no_grad():
                conv_formatted = rw_tokenizer.apply_chat_template(c, tokenize=False)
                conv_tokenized = rw_tokenizer(conv_formatted, return_tensors="pt").to(device)
                s = rw_model(**conv_tokenized).logits[0][0].item()
                scores.append(s)
            
        index = scores.index(max(scores))
        return responses[index]

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=60))
async def get_chat_completion_BoN(message: str, semaphore, N=4, retry_count=0) -> str:
    if dataset_name in ['usaco', 'livecodebench']:
        head_step = {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each time, you just need to generate a response contain one JSON resposne of current step. For each step, provide a title that describes what you're doing in that step, along with the content and related code that you can solve the problem. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', 'code', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

                Example of a valid JSON response:
                ```json
                {
                    "title": "Identifying Key Information",
                    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves....",
                    "code":"def fun(): ...",
                    "next_action": "continue"
                }```"""}
        
        head_choose = {"role": "system", "content": """You are an expert AI assistant and you need to choose a response based on the context, please just tell me the response in the valid JSON response.

                Example of a valid JSON response:
                ```json
                {
                    "title": "Identifying Key Information",
                    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves... .  And the related code for the solution is : ....",
                    "code":"def fun(): ...",
                    "next_action": "continue"
                }```"""}
    else:
        head_step = {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', 'code', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

                Example of a valid JSON response:
                ```json
                {
                    "title": "Identifying Key Information",
                    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
                    "next_action": "continue"
                }```"""}
        head_choose = {"role": "system", "content": """You are an expert AI assistant and you need to choose a response based on the context, please just tell me the response in the valid JSON response.

                Example of a valid JSON response:
                ```json
                {
                    "title": "Identifying Key Information",
                    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
                    "next_action": "continue"
                }```"""}
    try:
        async with semaphore:  # 使用传入的信号量限制并发
            steps = []
            step_count = 1
            messages =  [
                        head_step,
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
                    ]

            response_results_setp = []
            for i in range(N):
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=180
                )
                response_results_setp.append(response.choices[0].message.content)
            
            #选最优的结果
            history_prompts = []
            for item in messages:
                history_prompts.append(item['content'])
            if N == 1:
                response_result_setp = response_results_setp[0]
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        head_choose,
                        {"role": "user", "content": f'Give you a context: {history_prompts} and the responses if this context: {response_results_setp}. Please tell one which one response is the most correct for the context, just tell me the context of the response.'}
                        ],
                    timeout=180
                )
                response_result_setp = response.choices[0].message.content


            step_data = load_response_json(response_result_setp)

            if 'code' in step_data:
                steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], step_data['code']))
            else:
                steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'] ))


            while True:
                if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                    break
                
                messages.append({"role": "assistant", "content": json.dumps(step_data)})
                response_results_setp = []
                for i in range(N):
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=180
                    )
                    response_results_setp.append(response.choices[0].message.content)
                #选最优的结果

                history_prompts = []
                for item in messages:
                    history_prompts.append(item['content'])
                if N == 1:
                    response_result_setp = response_results_setp[0]
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            head_choose,
                            {"role": "user", "content": f'Give you a context: {history_prompts} and the responses: {response_results_setp}. Please tell one which one response is the most correct for the context, just tell me the context of the response.'}
                            ],
                        timeout=180
                    )
                    response_result_setp = response.choices[0].message.content

                step_data = load_response_json(response_result_setp)

                if 'code' in step_data:
                    steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], step_data['code']))
                else:
                    steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'] ))
                
                step_count += 1

            messages.append({"role": "assistant", "content": 'Please provide the final answer based solely on your reasoning above. Please reminder you just need to give the text which meet the initial requirement, and don\'t need extra introduction.'})
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=360
            )

            final_data = load_response_json(response.choices[0].message.content)
            if 'code' in final_data:
                steps.append((f"Step {step_count}: {final_data['title']}", final_data['content'], final_data['code']))
            else:
                steps.append((f"Step {step_count}: {final_data['title']}", final_data['content'] ))

            if dataset_name in ['usaco', 'livecodebench']:
                return {
                    'steps': steps,
                    'response_result': final_data['code'],
                    'message': message,
                }
            else:
                return {
                    'steps': steps,
                    'response_result': final_data['content'],
                    'message': message,
                }
        
    except Exception as e:
        print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
        raise


async def request_model(prompts, N):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async def wrapped_get_chat_completion(prompt):
        try:
            return await get_chat_completion_BoN(prompt, semaphore, N)
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
    parser = argparse.ArgumentParser(description="config for o1 capability analysis")

    parser.add_argument("--dataset_name" , type = str , default = 'hotpotqa')
    parser.add_argument("--model_name" , type = str , default = 'GPT4o')
    parser.add_argument("--N" , type = int , default = 1)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    N = args.N

    # dataset_name = 'hotpotqa'
    # dataset_name = 'collie'
    # model_name = 'GPT4o'
    # model_name = 'Claude'
    # model = "text-embedding-3-large"

    global rw_model
    global rw_tokenizer
    rw_model = None
    rw_tokenizer = None

    global model
    if model_name == 'GPT4o': 
        model = "gpt-4o-2024-08-06"
    elif model_name == 'Claude':
        model = 'claude-3-5-sonnet-20240620'
    
    if dataset_name == 'hotpotqa':
        all_data = load_json('./data/hotpotqa_sentence_bert_filter.json')#[:10]
    elif dataset_name == 'collie':
        all_data = load_dill('./data/collie_sentence_bert_filter.dill')#[10:15]#[10:20]
    elif dataset_name == 'aime':
        data_aimo = load_dataset('AI-MO/aimo-validation-aime')['train']#.select(range(4))
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
        responses = asyncio.run(request_model(prompts, N))
    elif dataset_name == 'collie':
        prompt2item = {}
        for item in all_data:
            content = item['prompt']
            prompts.append(content)
            prompt2item[content] = item
        responses = asyncio.run(request_model(prompts, N))
    elif dataset_name == 'aime':
        all_data = []
        prompt2item = {}
        for item in data_aimo:
            problem = item['problem']
            prompts.append(problem)
            all_data.append(item)
            prompt2item[problem] = item
        responses = asyncio.run(request_model(prompts, N))
    elif 'usaco' in dataset_name:
        prompt2item = {}
        for item in tqdm(all_data):
            messages = item['messages']
            content = messages[0]['content']
            prompts.append(content)
            prompt2item[content] = item
        responses = asyncio.run(request_model(prompts, N))
    
    #根据下标检索item
    results = []
    for response in responses:
        if response != None:
            response_result = response['response_result']
            steps = response['steps']
            prompt = response['message']
            item = prompt2item[prompt]
            item['response'] = response_result
            item['steps'] = steps
            results.append(item)
        else:
            results.append(None)

    os.makedirs(f'./results/', exist_ok=True)

    if dataset_name == 'collie':
        with open(f'./results/step_wise_BoN_{N}_{model_name}_{dataset_name}.dill', 'wb') as f:
            dill.dump(results, f)
    else:
        json_save(results, f'./results/step_wise_BoN_{N}_{model_name}_{dataset_name}.json')

    
