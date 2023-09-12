# Video Llama

## Preparation

Build the environment
`pip install -r requirements.txt`

`export PYTHONPATH="./:$PYTHONPATH"`

## Generating Features

Generate video features
```
python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path ./Breakfast/Videos/BreakfastII_15fps_qvga_sync/ \
        --clip_feat_path features_breakfast
```

## Generate Chat Data

Generate chat data
```
python scripts/convert_instruction_txt_to_training_format_breakfast.py \
        --input_json_file ./LLM/Breakfast/Breakfast/ \
        --output_json_file video_chatgpt_training.json
```

## Training

Train the model

```
torchrun --nproc_per_node=2 --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path mmaaz60/LLaVA-7B-Lightening-v1-1 \
          --version v1 \
          --data_path video_chatgpt_training.json \
          --video_folder features_breakfast \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 1 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 1000 \
          --save_total_limit 3 \
          --learning_rate 2e-4 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```

## Testing

Test the model to get the metrics

```
python video_chatgpt/eval/run_inference.py --model-name breakfast_finetuned_videochatgpt --projection_path Video-ChatGPT_7B-1.1_Checkpoints/model.bin 
```

## Inference 

You can run the chat-style inference
```
python video_chatgpt/demo/video_demo.py \
        --model-name 50salads_finetuned_videochatgpt \
        --projection_path Video-ChatGPT_7B-1.1_Checkpoints/model.bin
```