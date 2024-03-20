import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional
from transformers import AutoTokenizer
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM
# from peft import PeftModel
from video_chatgpt.constants import *


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class VideoChatGPTTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()


            ## NEW CODE ##
            try:
                model = model_to_save
            except:
                model = self.model
            # model.save_pretrained("50salads_finetuned_videochatgpt")
            # tokenizer = AutoTokenizer.from_pretrained('mmaaz60/LLaVA-7B-Lightening-v1-1')
            # model = VideoChatGPTLlamaForCausalLM.from_pretrained('mmaaz60/LLaVA-7B-Lightening-v1-1', low_cpu_mem_usage = True, torch_dtype =torch.float16)
            # mm_use_vid_start_end = True

            # # Add tokens to tokenizer
            # tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            # if mm_use_vid_start_end:
            #     tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

            # # Resize token embeddings of the model
            # model.resize_token_embeddings(len(tokenizer))
            # model = PeftModel.from_pretrained(model, '50salads_finetuned_videochatgpt')
            # model = model.merge_and_unload()
            # os.system('rm -r 50salads_finetuned_videochatgpt')
            # model.save_pretrained("50salads_finetuned_videochatgpt")
            ## NEW CODE ##

            model.save_pretrained("breakfast_finetuned_videochatgpt")
            weight_to_save = {}
            keys_to_match = ['positional_encodings', 'embed_tokens', 'norm', 'input_layernorm', 'post_attention_layernorm']
            for k, v in _state_dict.items():
                # if any(key_match in k for key_match in keys_to_match):
                weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "model")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'model.bin'), )

        # super(VideoChatGPTTrainer, self)._save(output_dir, state_dict)