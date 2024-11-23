# Fact-Checking Complex Claims with Program-Guided Reasoning

## Group Members
Group 7:
- 2220409: Dang Hoang Anh
- 2410407: The-Hai NGUYEN
- 2410408: Tan-Minh NGUYEN


## Re-implementation

### How to use
Install the required packages:
```bash
pip install -r requirements.txt
```

There are two main steps to use the code: Reasoning Program Generation and Reasoning Program Execution.

#### Reasoning Program Generation
abc

#### Reasoning Program Execution
Run the following command to execute the reasoning program:
```bash
bash run_program_execution.sh
```
By default, this script will use the `google/flan-t5-large` model to execute the [generated 
reasoning program](./results/programs/HOVER-4_N=1_meta-llama/Llama-3.1-8B-Instruct_programs_v3.json) from the `meta-llama/Llama-3.1-8B-Instruct` model. The results will be saved in the [`llama_3.1_8b_instruct_flan_t5_large.json`](./results/execution/llama_3.1_8b_instruct_flan_t5_large.json) file.

### Results

Reasoning/Execution models | Llama 3.1 8B | Llama 3.1 8B Instruct | Llama 3 70B Instruct 
--- | --- | --- | --- 
Flan T5 large | 61.22 | 62.82 | 59.46
Flan T5 xl | 61.04 | 61.11 | 59.15
