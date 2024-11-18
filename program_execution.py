import pandas as pd
import torch
from tqdm import tqdm
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_dataset(path):
    return pd.read_json(path, lines=True, orient="records")


class T5QuestionAnswering:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def generate(self, input_string, **generator_args):
        device = self.model.device
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(device)
        with torch.no_grad():
            res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    
    def answer_verify_question(self, claim, evidence):
        input_text = f"{evidence}\nBased on the above information, is it true that {claim}? True or false? The answer is: "

        return self.generate(input_text, 
                            max_length = None, 
                            max_new_tokens=8)[0].strip()


    def answer_question_directly(self, question, evidence):
        input_text = f"{evidence}\nQuestion: {question}\nThe answer is:"

        return self.generate(input_text, 
                            max_length = None, 
                            max_new_tokens=32)[0].strip()


class ProgramExecution:
    def __init__(self, args):
        self.args = args
        self.QA_module = T5QuestionAnswering(args.QA_model_name)


    def parse_verify_command(self, command, variable_map):
        return_var, tmp = command.split('= Verify')
        return_var = return_var.strip()
        # claim = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Verify\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        claim = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if claim.find(replace_var) >=0:
                claim = claim.replace(replace_var, variable_value)

        return return_var, claim


    def parse_question_command(self, command, variable_map):
        return_var, tmp = command.split('= Question')
        return_var = return_var.strip()
        # question = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Question\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        question = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if question.find(replace_var) >=0:
                question = question.replace(replace_var, variable_value)

        return return_var, question

    
    def derive_final_answer(self, command, variable_map):
        final_label = True

        command = command.replace('label =', '').strip()
        p1 = re.compile(r'Predict[(](.*?)[)]', re.S)
        command_arg = re.findall(p1, command)[0]
        verify_subs = command_arg.split(" and ")
        arguments = [arg.strip() for arg in verify_subs]
        for argument in arguments:
            if argument in variable_map:
                final_label = variable_map[argument] and final_label
            else:
                print(f"Alert!!! wrong argument: {argument}")
        
        return final_label


    def map_direct_answer_to_label(self, predict):
        predict = predict.lower().strip()
        label_map = {'true': True, 'false': False, 'yes': True, 'no': False, "it's impossible to say": False}
        if predict in label_map:
            return label_map[predict]
        else:
            print(f"Alert!!! wrong answer mapping: {predict}")


    def parse_program(self, id, program, evidence):
        final_answer = None
        
        variable_map = {}   
        for command in program:
            c_type = self.get_command_type(command)
            

            if c_type == "VERIFY":
                return_var, claim = self.parse_verify_command(command, variable_map)
                answer = self.QA_module.answer_verify_question(claim, evidence)
                variable_map[return_var] = self.map_direct_answer_to_label(answer)
            elif c_type == "QUESTION":
                return_var, question = self.parse_question_command(command, variable_map)
                answer = self.QA_module.answer_question_directly(question, evidence)
                variable_map[return_var] = answer
            elif c_type == "PREDICT":
                final_answer = self.derive_final_answer(command, variable_map)

        return final_answer


    def get_command_type(self, command):
        if command.find("= Predict")>=0:
            return "PREDICT"
        elif command.find('= Verify')>=0:
            return "VERIFY"
        elif command.find('= Question')>=0:
            return "QUESTION"
        else:
            return "UNKNOWN"


    def execute_on_dataset(self):
        reasoning_program_df = load_dataset(self.args.reasoning_program_path)
        raw_df = load_dataset(self.args.dataset_path)

        results = []
        for _, sample in tqdm(reasoning_program_df.iterrows()):
            program = sample['predicted_programs'][0][0]
            evidence = raw_df[raw_df['id'] == sample['id']]['evidence'].values[0]

            # execute program
            sample_prediction = self.parse_program(sample['id'], program, evidence)

            results.append({'id': sample['id'], 
                            'claim': sample['claim'],
                            'gold': sample['gold'], 
                            'prediction': 'supports' if sample_prediction == True else 'refutes'})

        results_df = pd.DataFrame(results)
        results_df.to_json(self.args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    #TODO: Remove
    from dotted_dict import DottedDict
    args = DottedDict({
        "reasoning_program_path": "samples.jsonl",
        "dataset_path": "dev.jsonl",
        "output_path": "hehe.jsonl",
        "QA_model_name": "google/flan-t5-large"
    })
    

    program_execution = ProgramExecution(args)
    program_execution.execute_on_dataset()