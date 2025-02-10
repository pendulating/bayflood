# %%
from glob import glob 
import pandas as pd 
import numpy as np 

from random import shuffle

# %%
entire_sep29 = glob("/scratch/datasets/all_no_letterboxing/*.jpg")
len(entire_sep29)

# %%
N = 6 
# split into N groups
shuffle(entire_sep29)

groups = np.array_split(entire_sep29, N)
# print length of each group
for i in range(N):
    print(i, len(groups[i]))


# %%
# write each group to csv 
for i in range(N):
    df = pd.DataFrame(groups[i], columns=['image_path'])
    df.to_csv(f"entire_sep29_{i}.csv", index=False)

# %%



