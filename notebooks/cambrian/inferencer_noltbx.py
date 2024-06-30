# %%
import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
from glob import glob 
import pandas as pd 
import os

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



# %%

# cambrian-phi3-3b
# conv_mode = "phi3"

# cambrian-8b
#conv_mode = "llama_3" 

# cambrian-34b
#conv_mode = "chatml_direct"

# cambrian-13b
conv_mode = "vicuna_v1"

def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = os.path.expanduser("nyu-visionx/cambrian-13b")
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

temperature = 0

# %%
anl_set_jun16_w_extra_from_original = glob("../../datasets/anl_set_jun16_w_extra_from_original/*/*/*.jpg")




print(len(anl_set_jun16_w_extra_from_original))
# turn into dataframe 
anl_set_jun16_w_extra_from_original_df = pd.DataFrame(anl_set_jun16_w_extra_from_original, columns=["image_path"])

anl_set_jun16_w_extra_from_original_df["split"] = anl_set_jun16_w_extra_from_original_df["image_path"].apply(lambda x: x.split("/")[-3])
anl_set_jun16_w_extra_from_original_df["class"] = anl_set_jun16_w_extra_from_original_df["image_path"].apply(lambda x: x.split("/")[-2])

# replace path except basename with no_ltbx_path 
no_ltbx_path = '/share/ju/nexar_data/training_datasets/street_flooding/all_no_letterboxing'

anl_set_jun16_w_extra_from_original_df["image_path"] = anl_set_jun16_w_extra_from_original_df["image_path"].apply(lambda x: os.path.join(no_ltbx_path, "nlbx_"+os.path.basename(x)))

anl_set_jun16_w_extra_from_original_df["q0"] = "Does this image show a flooded street?"
anl_set_jun16_w_extra_from_original_df["q1"] = "Does this image show more than a foot of standing water?"
anl_set_jun16_w_extra_from_original_df["q2"] = "Is the street in this image flooded?"
anl_set_jun16_w_extra_from_original_df["q3"] = "Could a car drive through the water in this image?"
anl_set_jun16_w_extra_from_original_df["q4"] = "Does this image show a visible street?"
anl_set_jun16_w_extra_from_original_df["q5"] = "Is there any visible street in this image?"
anl_set_jun16_w_extra_from_original_df["q6"] = "Is the view from windshield in this image too obstructed?"


anl_set_jun16_w_extra_from_original_df 



# %%
for index, row in tqdm(anl_set_jun16_w_extra_from_original_df.iterrows()):

    for i in range(7):

        image_path = row["image_path"]
        image = Image.open(image_path).convert('RGB')
        question = row["q" + str(i)]

        input_ids, image_tensor, image_sizes, prompt = process(image, question, tokenizer, image_processor, model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        anl_set_jun16_w_extra_from_original_df.loc[index, "response_" + str(i)] = outputs

        # write to csv in case of crash
        anl_set_jun16_w_extra_from_original_df.to_csv("NO_LTBX_anl_set_jun16_w_extra_from_original_df.csv")

# %%



