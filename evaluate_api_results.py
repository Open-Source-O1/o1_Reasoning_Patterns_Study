import dill, argparse, pickle
from tqdm import tqdm
from utils import load_json, json_save, load_dill

def response_replace(response, text):
    if text in response:
        response = response.replace(text, '')
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for MMRA benchmark")

    parser.add_argument("--dataset_name" , type = str , default = 'hotpotqa')
    # parser.add_argument("--model_name" , type = str , default = 'o1')
    parser.add_argument("--model_name" , type = str , default = 'GPT4o')
    parser.add_argument("--BoN_type" , type = str , default = None)  # BoN, step_wise_BoN
    parser.add_argument("--N" , type = int , default = 4)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    BoN_type = args.BoN_type
    N = args.N

    print(model_name)
    if 'o1' in model_name:
        if 'collie' in dataset_name:
            with open(f'./results/{model_name}_{dataset_name}.pkl', 'rb') as f:
                data = pickle.load(f)
            save_path = f'./results/{model_name}_{dataset_name}_judge.pkl'
        else:
            data = load_json(f'./results/{model_name}_{dataset_name}.json')
            save_path = f'./results/{model_name}_{dataset_name}_judge.json'


    if dataset_name == 'hotpotqa':
        if BoN_type != None:
            data = load_json(f'./results/{BoN_type}_{N}_{model_name}_hotpotqa.json')
            save_path = f'./results/{BoN_type}_{N}_{model_name}_hotpotqa_judge.json'
        else:
            data = load_json(f'./results/{model_name}_hotpotqa.json')
            save_path = f'./results/{model_name}_hotpotqa_judge.json'
    elif dataset_name == 'aime':
        if BoN_type != None:
            data = load_json(f'./results/{BoN_type}_{N}_{model_name}_aime.json')
            save_path = f'./results/{BoN_type}_{N}_{model_name}_aime_judge.json'
        else:
            data = load_json(f'./results/{model_name}_aime.json')
            save_path = f'./results/{model_name}_aime_judge.json'
    elif dataset_name == 'collie':
        if BoN_type != None:
            data = load_dill(f'./results/{BoN_type}_{N}_{model_name}_collie.dill')
            save_path = f'./results/{BoN_type}_{N}_{model_name}_collie_judge.dill'
        else:
            data = load_dill(f'./results/{model_name}_collie.dill')
            save_path = f'./results/{model_name}_collie_judge.dill'

    count = 0
    right = 0
    results = []
    if dataset_name == 'hotpotqa' or dataset_name == 'aime':
        for item in tqdm(data):

            count += 1
            if item != None:
                response = item['response'].lower()
                answer = item['answer'].lower()
                
                if answer in response:
                    item['judge'] = 1
                    right += 1
                else:
                    item['judge'] = 0
                results.append(item)
        json_save(results, save_path)

    elif dataset_name == 'collie':
        for item in tqdm(data ):
            count += 1
            if item != None:
                constraints = item['constraint']
                targets = item['targets']
                response = item['response']
                
                if type(response) == type(''):
                    judge = constraints.check(response, targets)

                    if judge == 1:
                        right += 1
                    
                    item['judge'] = judge
                    results.append(item)

        with open(save_path, 'wb') as f:
            dill.dump(results, f)

    print(right, '/', count)
    print('percentage %:', round(right/ count, 4) * 100 )
