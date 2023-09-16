# Video Llama

## Preparation

Build the environment
`pip install -r requirements.txt`

`export PYTHONPATH="./:$PYTHONPATH"`

## Generating Features

Generate video features by following command. The model works on the video features. Since the video features extractor is frozen and is not trained so we extract the features before training to avoid any overhead during training.

```
python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path ./Breakfast/Videos/BreakfastII_15fps_qvga_sync/ \
        --clip_feat_path features_breakfast
```

The code is modified according to breakfast structure. if you want to train on another dataset, you have to modify the nested loops (line 111-113) according to the dataset structure of your dataset.
This will generate a new folder `features_breakfast`. you just need to generate it once and then you can skip this step in future unless you want to modify how the features are being extracted or how they are being stored.
Currently the features from each layer is saved, if you want to save features from only the last layer then you have to modify the code accordingly.

Note: Currently the features extracted are for original videos. if you want to extract features for augmentations as well, then comment line 116-120.

## Generate Chat Data

Generate chat data for the model to train by following command. This json file will be used during training. the format is simple, it contains corresponding video, the questions and their answers in a chat-style format.
The videos are being repeated to increase the size of dataset, we have tried two approaches, one is to replicate the existing videos and other is to augment the existing videos. Currently its following the first approach, if you want to use the second one, first make sure you have extracted augmented videos features as well and then you will have to uncomment line 126-129 and comment line 130 in order to use augmented features.
```
python scripts/convert_instruction_txt_to_training_format_breakfast.py \
        --input_json_file ./LLM/Breakfast/Breakfast/ \
        --output_json_file video_chatgpt_training.json
```

Repeatation is very import parameter, you can't have too many or too few repeatations/augmentations. experiments with number repeatations on line 123-124 to get an optimal number.
Currently its set for 12 repeats for segmentations questions and 2 repeats for other questions.

Similar to previous command, the code is according to the breakfast dataset structure. If you have new dataset, modify the code accordingly.

## Training

Train the model. Most of the training parameters can be modified in below command.

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

If you want to change the model architecture, please refer to `video_chatgpt/model/video_chatgpt.py` and start with line number 226. `VideoChatGPTLlamaForCausalLM` contains the overall and high level view of whole model's architecture. You can go to each of the module one by one to make any changes.

For example: I had to add positional encodings to input video features, I navigated to `VideoChatGPTLlamaForCausalLM` first and then I found that it directly send all the features to `VideoChatGPTLlamaModel` module so I can't add the positional encoding there and I have to go one level deeper. So I added positional encoding to the module `VideoChatGPTLlamaModel` and modified its `forward` function to make sure where these are being applied.
some times you might have to go to even deeper modules, for example the `VideoChatGPTLlamaModel` module has parent class `LlamaModel` where the main pretrained modules of self attention are applied. I had to apply cross attention so I had to go to `LlamaModel` and add my custom module and apply them there. By the way, the whole `LlamaModel` is defined in file `video_chatgpt/model/modeling_llama.py` this was originally `transformers` library's file but I copied it in our repo so that we can make changes to it. In future if you want to do something similar, don't be afraid to override the library files (don't modify library files directly just copy paste them in your repo and make the import and function calls accordingly in your code)

## Testing

Test the model to get the metrics. 

```
python video_chatgpt/eval/run_inference.py --model-name breakfast_finetuned_videochatgpt --projection_path Video-ChatGPT_7B-1.1_Checkpoints/model.bin 
```

If you want to add some new metric, please modify the `video_chatgpt/eval/metrics.py` file.

## Inference 

You can run the chat-style inference
```
python video_chatgpt/demo/video_demo.py \
        --model-name 50salads_finetuned_videochatgpt \
        --projection_path Video-ChatGPT_7B-1.1_Checkpoints/model.bin
```