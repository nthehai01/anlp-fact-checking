# Fact-Checking Complex Claims with Program-Guided Reasoning

## Group Members
Group 7:
- 2220409: Dang Hoang Anh
- 2410407: The-Hai NGUYEN
- 2410408: NGUYEN Tan Minh


## Re-implementation

### How to use
Install the required packages:
```bash
pip install -r requirements.txt
```

There are two main steps to use the code: Reasoning Program Generation and Reasoning Program Execution.

#### Reasoning Program Generation
Run the following command to generate the reasoning program:
```bash
python program_generator.py \
    --data_path ./datasets \
    --dataset_name "HOVER-4" \
    --num_programs_per_example "1" \
    --model_name meta-llama/Llama-3.1-8B-Instruct\
    --num_eval_samples "1039" \
    --api_key "Your HF access token to model" \
    --save_path ./results/programs
```
The default setting for program_generator is `meta-llama/Llama-3.1-8B-Instruct` model to generate reasoning programs from 1039 samples HOVER 4-hop dataset. The number of different reasoning paths is 1. 

#### Reasoning Program Execution
Run the following command to execute the reasoning program:
```bash
python program_execution.py\
    --reasoning_program_path results/programs/HOVER-4_N=1_meta-llama/Llama-3.1-8B-Instruct_programs_v3.json\
    --dataset_path datasets/HOVER-4/claims/dev.json\
    --output_path results/executions/llama_3.1_8b_instruct_flan_t5_large.json\
    --QA_model_name google/flan-t5-large
```
By default, this script will use the `google/flan-t5-large` model to execute the [generated 
reasoning program](./results/programs/HOVER-4_N=1_meta-llama/Llama-3.1-8B-Instruct_programs_v3.json) from the `meta-llama/Llama-3.1-8B-Instruct` model. The results will be saved in the [`llama_3.1_8b_instruct_flan_t5_large.json`](./results/executions/llama_3.1_8b_instruct_flan_t5_large.json) file.

### Results

Reasoning/Execution models | Llama 3.1 8B | Llama 3.1 8B Instruct | Llama 3 70B Instruct 
--- | --- | --- | --- 
Flan T5 large | 61.22 | 62.82 | 59.46
Flan T5 xl | 61.04 | 61.11 | 59.15
