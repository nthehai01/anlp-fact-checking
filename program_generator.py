import argparse
import os
import json
from tqdm import tqdm
import torch

from prompts import Prompt_Loader
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

# token to access Llama models on HF
access_token = "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"


class MyStoppingCriteria(StoppingCriteria):
    """ Define stopping generation conditions. """
    def __init__(self, target_sequence, prompt, tokenizer):
        self.target_sequence = target_sequence
        self.prompt=prompt
        self.tokenizer=tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class Reasoning_Program_Generator:
    """ LLMs-based program generator module. """
    
    def __init__(self, args):
        """ Initialize the LLM by using Huggingface library.
            Model is loaded with torch.bfloat16 precesision, flash attention and KV attention cache.
            
            Args:
                data_path (str): path to the dataset
                data_name (str): name of the dataset
                model_name (str): name of the LLM
                save_path (str): path to the output folder
                num_programs_per_example (str): number of generated reasoning programs
            Returns:
                (str): Generated reasoning program
        """
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example
        
        self.gen_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation='flash_attention_2')
        self.gen_model.generation_config.cache_implementation = "static"

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompt_loader = Prompt_Loader()

    def update_results(self, sample, generated_text):
        """Update generated programs.
        A generated program is a list, with each element is a operator (sub-task).
        
            Args:
                generated_text (list): Generated program
                
            Returns:
        """
        program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]
        self.result_dict[sample['id']]['predicted_programs'].append(program_list)

    def batch_generate_programs(self, batch_size = 10):
        """ Generate the reasoning program using LLMs.
        
            Args:
            
            Returns:
        """
        
        # create output_dir
        self.result_dict = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataset
        with open(os.path.join(self.data_path, self.dataset_name, 'claims', 'dev.json'), 'r') as f:
            raw_dataset = json.load(f)
        
        raw_dataset = raw_dataset if self.args.num_eval_samples < 0 else raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} dev set.")

        # generate programs
        temperature = 0.000001 if self.num_programs_per_example == 1 else 0.7
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        
        # initialize empty results
        result_dict = {}
        for idx, sample in enumerate(raw_dataset):
            result = {'idx': idx,
                        'id': sample['id'], 
                        'claim': sample['claim'],
                        'gold': sample['label'], 
                        'predicted_programs': []}
            result_dict[sample['id']] = result
        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            print(f"Generating programs for iteration {iteration + 1}...")
            # for each chunk
            for chunk in tqdm(dataset_chunks):
                # create prompt
                full_prompts = [self.prompt_loader.prompt_construction(example['claim'], self.dataset_name) for example in chunk]
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        model_input = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
                        generated_ids = self.gen_model.generate(**model_input, max_new_tokens=args.max_new_tokens, temperature=temperature, stopping_criteria=MyStoppingCriteria(args.stop_words, full_prompt, self.tokenizer))
                        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        self.update_results(sample, output)
                    except:
                        print('Error in generating reasoning programs for example: ', sample['id'])

        print(f"Generated {len(result_dict)} examples.")
        # create outputs
        for key in result_dict:
            outputs.append(result_dict[key])
        sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

        # save outputs
        os.makedirs(os.path.dirname(os.path.join(self.save_path, f'{self.dataset_name}_N={self.num_programs_per_example}_{self.model_name}_programs.json')), exist_ok=True)
        with open(os.path.join(self.save_path, f'{self.dataset_name}_N={self.num_programs_per_example}_{self.model_name}_programs.json'), 'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    """ Receive input arguments for experiment settings."""
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='HOVER', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default = './results/programs', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--stop_words', type=str, default='# The claim is')
    parser.add_argument('--max_new_tokens', type=int, default=256) # 1024 -> 256
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generator = Reasoning_Program_Generator(args)
    generator.batch_generate_programs()