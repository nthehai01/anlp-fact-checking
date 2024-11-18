echo $(pwd)
WORKSPACE=$(pwd)
PYTHONPATH=$WORKSPACE python ./models/program_generator.py \
    --data_path ./datasets \
    --dataset_name "HOVER-4" \
    --num_programs_per_example "1" \
    --model_name meta-llama/Meta-Llama-3-70B-Instruct\
    --num_eval_samples "1039" \
    --api_key "Your OpenAI API Key" \
    --save_path ./results/programs