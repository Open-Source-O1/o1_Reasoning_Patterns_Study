# o1_Reasoning_Patterns_Study

This is the repo for the paper [A Comparative Study on Reasoning Patterns of OpenAI's o1 Model](https://arxiv.org/pdf/2410.13639).

<div align="center">
<img src=./figures/main_result.png width=70% />
</div>

Enabling Large Language Models (LLMs) to handle a wider range of complex tasks (e.g., coding, math) has drawn great attention from many researchers. 
As LLMs continue to evolve, increasing the number of model parameters yields diminishing performance improvements and heavy computational costs.
Recently,
OpenAI's o1 model has shown that inference strategies (i.e., Test-time Compute methods) can also significantly enhance the reasoning capabilities of LLMs. However, the mechanisms behind these methods are still unexplored.
In our work, to investigate the reasoning patterns of o1, we compare o1 with existing Test-time Compute methods (BoN, Step-wise BoN, Agent Workflow, and Self-Refine) by using OpenAI's GPT-4o as a backbone on general reasoning benchmarks in three domains (i.e., math, code and commonsense reasoning).
Specifically,
first,
our experiments show that the o1 model has achieved the best performance on most datasets.
Second,
as for the methods of searching diverse responses (e.g., BoN), we find the reward models' capability and the search space both limit the upper boundary of these methods.
Third,
as for the methods that break the problem into many sub-problems, the Agent Workflow has achieved better performance than Step-wise BoN due to the domain-specific system prompt for planning better reasoning processes.
Fourth,
we summarize six reasoning patterns of o1 and provide a detailed analysis across different reasoning benchmarks.

<div align="center">
<img src=./figures/pattern_analysis.png width=60% />
</div>

<div align="center">
<img src=./figures/result.png width=60% />
</div>

## Dataset
Our filtered data can be found in this repo: [data](https://github.com/Open-Source-O1/o1_Reasoning_Patterns_Study/tree/main/data).

Besides, we also upload those data into hf dataset. It can be used by the following code:

```python
from datasets import load_dataset
data = load_dataset('SiweiWu/o1_Reasoning_Patterns_Study')
```

However, apart from the text data, Collie also provides functions that can be directly used to evaluate the generated responses of LLMs, but it cannot be uploaded to the HF dataset. Therefore, for the Collie dataset, we still suggest that you load the data from the GitHub folder [data](https://github.com/Open-Source-O1/o1_Reasoning_Patterns_Study/tree/main/data).


## Running
You can run the run.sh to reproduce the results on our main table.

```python
bash run.sh
```

You can change the '--N' and 'step_wise_type' and 'dataset_name' to obtain the results at different settings.

As for the evaluation of the Code task result (USACO), we are organizing this part of the code.
As for Self-Refine, we use the code from [Self-Refine](https://github.com/madaan/self-refine).
As for the Agent Workflow, we utilize the state-of-the-art agent framework [Agents 2.0: Symbolic Learning Enables Self-Evolving Agents](https://github.com/aiwaves-cn/agents) on the HotpotQA and Collie, and we use the [GPTs](https://openai.com/index/introducing-gpts/) for USACO and AIME.

## Results in Our Paper

We will release the output files of our main results soon ...

## Citation

```python
@article{qu2024overview,
  title={Overview of the NLPCC 2024 Shared Task on Chinese Metaphor Generation},
  author={Qu, Xingwei and Zhang, Ge and Wu, Siwei and Li, Yizhi and Lin, Chenghua},
  journal={arXiv preprint arXiv:2408.04378},
  year={2024}
}
```
