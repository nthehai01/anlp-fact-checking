import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_dataset(path):
    """ Load the dataset from the given path.
        The dataset is assumed to be in JSON format.
        
        Args:
            path (str): Path to the dataset file
        Returns:
            (pd.DataFrame): DataFrame containing the dataset
    """

    with open(path, 'r') as file:
        data = json.load(file)

    return pd.DataFrame(data)


class T5QuestionAnswering:
    """ T5 Question Answering module. """

    def __init__(self, model_name):
        """ Initialize the T5 Question Answering module by using AutoModelForSeq2SeqLM.
            Model is loaded with torch.bfloat16 precision. 
        
            Args:
                model_name (str): Name of the T5 model
        """

        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def generate(self, input_string, **generator_args):
        """ Generate the answer for the given input string. The input string is first
            tokenized and truncated to the max length of the model of 512 tokens
            (if there is any warning about the input length, you can ignore it).
            The tokenized input is then passed to the model for generation. The generated
            tokens are decoded using the tokenizer to get the answer.

            Args:
                input_string (str): Input string
                generator_args (dict): Arguments for the generator
            Returns:
                (str): Generated answer
        """

        device = self.model.device
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(device)
        input_ids = input_ids[-512:]  # Get the last 512 tokens since Flan T5 has a max length of 512

        with torch.no_grad():
            res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    
    def answer_verify_question(self, claim, evidence):
        """ Answer the VERIFICATION question based on the given claim and evidence.
            In this case, we only use the gold evidence setting to generate the answer.
            Gold evidence is the evidence that is enough to aid the model to answer 
            the question without any need for retrieval.

            The prompt is as the `input_text` below.

            Args:
                claim (str): Claim
                evidence (str): Evidence
            Returns:
                (str): Generated answer
        """

        input_text = f"{evidence}\nBased on the above information, is it true that {claim}? True or false? The answer is: "

        return self.generate(input_text, 
                            max_length = None, 
                            max_new_tokens=8)[0].strip()


    def answer_question_directly(self, question, evidence):
        """ Answer the QA question based on the given claim and evidence.
            In this case, we also use the gold evidence setting to generate the answer.

            The prompt is as the `input_text` below.

            Args:
                claim (str): Claim
                evidence (str): Evidence
            Returns:
                (str): Generated answer
        """

        input_text = f"{evidence}\nQuestion: {question}\nThe answer is:"

        return self.generate(input_text, 
                            max_length = None, 
                            max_new_tokens=32)[0].strip()


class ProgramExecution:
    """ Main module for executing the generated reasoning programs. """

    def __init__(self, args):
        self.args = args
        self.QA_module = T5QuestionAnswering(args.QA_model_name)


    def parse_verify_command(self, command, variable_map):
        """ Parse the VERIFY command. The variable_map is used to replace the 
            variables in the claim with their values.

            For example, if the command is 'fact_1 = Verify("{answer_1} is LLM.")' 
            and the variable_map is {'answer_1': 'ChatGPT'}, then the returned 
            variable's name is 'fact_1' and the claim is 'ChatGPT is LLM.'

            Args:
                command (str): VERIFY command
                variable_map (dict): Mapping of variables to their values
            Returns:
                (str, str): variable's name and the claim
        """

        return_var, tmp = command.split('= Verify')
        return_var = return_var.strip()

        p1 = re.compile(f'Verify\\([f]?\"(.*)\"\\)', re.S)
        matching = re.findall(p1, command)
        claim = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if claim.find(replace_var) >=0:
                # TODO: check variable_value for sample 205
                variable_value = str(variable_value).lower()
                claim = claim.replace(replace_var, variable_value)

        return return_var, claim


    def parse_question_command(self, command, variable_map):
        """ Parse the Question command. The variable_map is used to replace the 
            variables in the question with their values.

            For example, if the command is 'answer_2 = Question("What is {answer_1}?")' 
            and the variable_map is {'answer_1': 'ChatGPT'}, then the returned 
            variable's name is 'answer_2' and the question is 'What is ChatGPT?'

            Args:
                command (str): QUESTION command
                variable_map (dict): Mapping of variables to their values
            Returns:
                (str, str): variable's name and the claim
        """

        return_var, tmp = command.split('= Question')
        return_var = return_var.strip()

        p1 = re.compile(f'Question\\([f]?\"(.*)\"\\)', re.S)
        matching = re.findall(p1, command)
        question = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if question.find(replace_var) >=0:
                question = question.replace(replace_var, variable_value)

        return return_var, question

    
    def derive_final_answer(self, command, variable_map):
        """ Derive the final answer based on the PREDICT command. The variable_map
            is used to replace the variables in the question with their values.
            
            For the sake of simplicity, this function supports only the 'and' & 'not'
            operators in the command, e.g., 'label = Predict(fact_1 and not fact_2)'. 
            Hence, commands like 'label = Predict(fact_1 or not fact_2)' are not
            supported yet.
            
            Args:
                command (str): PREDICT command
                variable_map (dict): Mapping of variables to their values
            Returns:
                (bool): Final label
            """


        final_label = True

        command = command.replace('label =', '').strip()
        p1 = re.compile(r'Predict[(](.*?)[)]', re.S)
        command_arg = re.findall(p1, command)[0]
        verify_subs = command_arg.split(" and ")
        arguments = [arg.strip() for arg in verify_subs]
        for argument in arguments:
            if argument in variable_map:
                final_label = variable_map[argument] and final_label
            elif argument.startswith("not"):  # TODO: check this for sample 236, 263, 478, 535, 628, 876
                argument = argument.replace("not", "").strip()
                final_label = (not variable_map[argument]) and final_label
            else:
                print(f"Alert!!! wrong argument: {argument}")
        
        return final_label


    def map_direct_answer_to_label(self, predict):
        """ Map the direct answer to the final verdict.
            positive answers -> True (supports)
            negative & unable answers & -> False (refutes)
        
            Args:
                predict (str): Direct answer
            Returns:
                (bool): Verdict
        """

        predict = predict.lower().strip()
        label_map = {
            'true': True, 
            'false': False, 
            'yes': True, 
            'no': False, 
            "it's impossible to say": False,
            'unanswerable': False,  # TODO: check this for sample 590
        }
        if predict in label_map:
            return label_map[predict]
        else:
            print(f"Alert!!! wrong answer mapping: {predict}")


    def parse_program(self, program, evidence):
        """ Parse the reasoning program into commands and execute them one by one.
            
            Args:
                program (list): List of commands
                
            Returns:
                (bool): Verdict label
                (dict): Mapping of variables to their values (for the purpose of debugging)
        """

        final_answer = None
        variable_map = {}

        for command in program:
            command = command.replace("= predict", "= Predict")  # fix bug for llama 3.1 8B

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

        return final_answer, variable_map


    def get_command_type(self, command):
        """ Get the type of the command based on the command string.
            Example commands:
                - 'label = Predict(fact_1 and not fact_2)'
                - 'fact_1 = Verify("{answer_1} is LLM.")'
                - 'answer_2 = Question("What is {answer_1}?")'

            Args:
                command (str): Command string
            Returns:
                (str): Command type
        """

        if command.find("= Predict") >= 0:
            return "PREDICT"
        elif command.find('= Verify') >= 0:
            return "VERIFY"
        elif command.find('= Question') >= 0:
            return "QUESTION"
        else:
            print(f"Alert!!! wrong command type: {command}")
            return "UNKNOWN"


    def execute_on_dataset(self):
        """ Execute the reasoning programs on the dataset. 
        
            The reasoning programs are loaded from the file and executed sample 
            by sample. The results are saved to a JSON-formatted file at the path
            specified in the args.output_path.
            
            The output file contains the following columns:
                - id: The id of the sample
                - claim: The claim of the sample
                - gold: The groundtruth label of the sample
                - prediction: The predicted verdict label of the sample
                - reasoning_variable_map: The mapping of variables to their values
        """

        reasoning_program_df = load_dataset(self.args.reasoning_program_path)
        raw_df = load_dataset(self.args.dataset_path)

        results = []
        for _, sample in tqdm(reasoning_program_df.iterrows()):
            program = sample['predicted_programs'][0]

            evidence = raw_df[raw_df['id'] == sample['id']]['evidence'].values[0]

            # execute program
            sample_prediction, variable_map = self.parse_program(program, evidence)

            results.append({'id': sample['id'], 
                            'claim': sample['claim'],
                            'gold': sample['gold'], 
                            'prediction': 'supports' if sample_prediction == True else 'refutes',
                            'reasoning_variable_map': variable_map})

        results_df = pd.DataFrame(results)
        results_df.to_json(self.args.output_path, orient='records')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--reasoning_program_path', 
        type=str,
        help='Path to generated reasoning programs file'
    )
    parser.add_argument(
        '--dataset_path', 
        type=str,
        help='Path to dataset file that contains evidences'
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        help='Path to save the output file'
    )
    parser.add_argument(
        '--QA_model_name', 
        type=str,
        default='google/flan-t5-xl',
        help='Name of the QA model'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    program_execution = ProgramExecution(args)
    program_execution.execute_on_dataset()
