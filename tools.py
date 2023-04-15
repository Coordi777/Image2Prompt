# import json
# import os
# import unicodedata
# import pandas as pd
# from tqdm.auto import tqdm
# from PIL import Image
#
#
# def is_english_only(string):
#     for s in string:
#         cat = unicodedata.category(s)
#         if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
#             return False
#     return True
#
#
# path = 'SD_1M.json'
# with open(path) as f:
#     dics = json.load(f)
#
# lis_pair = list(dics.items())
#
# df = pd.DataFrame(lis_pair, columns=['image_path', 'prompt'])
# df['prompt'] = df['prompt'].str.strip()
# df = df[df['prompt'].map(lambda x: len(x.split())) >= 5]
# df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
# df = df[df['prompt'].apply(is_english_only)]
# df['head'] = df['prompt'].str[:15]
# df['tail'] = df['prompt'].str[-15:]
# df.drop_duplicates(subset='head', inplace=True)
# df.drop_duplicates(subset='tail', inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.drop('head', axis='columns')
# df.drop('tail',axis='columns')
# dict_clean = dict(zip(df['image_path'],df['prompt']))
#
# image_names = list(dict_clean.keys())
# #
# # for image_name in tqdm(image_names):
# #     temp = Image.open(image_name).convert('RGB')
# #     if temp.size != (512, 512):
# #         dict_clean.pop(image_name)
#
# lis_pair = list(dict_clean.items())
# df2 = pd.DataFrame(lis_pair, columns=['image_path', 'prompt'])
# df2.to_csv('diffusiondb.csv', index=False)
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

path = 'metadata_withcmt.parquet'
dataframe = pd.read_parquet(path)
prompt_list = list(dataframe[dataframe['remove_idx_95']]['prompt'])
lengths = [len(word.split()) for word in prompt_list]
counter = Counter(lengths)
plt.bar(counter.keys(), counter.values())
plt.savefig('lengths.png')
print(counter)
