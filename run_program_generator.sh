python ./models/program_generator.py \
    --data_path ./datasets \
    --dataset_name "HOVER-4" \
    --num_programs_per_example "1" \
    --model_name meta-llama/Llama-3.1-8B-Instruct\
    --num_eval_samples "1039" \
    --api_key "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" \
    --save_path ./results/programs
