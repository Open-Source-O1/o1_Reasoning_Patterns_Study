python infer_api_model.py --dataset_name hotpotqa
python evaluate_api_results.py --dataset_name hotpotqa

python BoN.py --dataset_name hotpotqa --N 1
python evaluate_api_results.py --dataset_name hotpotqa --N 1 --BoN_type BoN

python step_wise_BoN.py  --dataset_name hotpotqa --N 1
python evaluate_api_results.py --dataset_name hotpotqa --N 1 --BoN_type step_wise_BoN