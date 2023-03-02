'''
计算余弦相似度
注意csv文件内的图片名字的顺序要对应好！！！！
'''
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from scipy import spatial

gt_path = '/Path/to/official/prompts.csv'
gt_prompts = pd.read_csv(gt_path)['prompt'].values.tolist()
save_gt = True
eIds = list(range(384))
images = os.listdir('/Path/to/official/images')
imgIds = [i.split('.')[0] for i in images]
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(imgIds, 384),
        np.tile(range(384), len(imgIds)))]
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
prompt_embeddings = st_model.encode(gt_prompts).flatten()
if save_gt:
    gt_csv = pd.DataFrame(
        index=imgId_eId,
        data=prompt_embeddings,
        columns=['val']
    ).rename_axis('imgId_eId')

    gt_csv.to_csv('ground_truth.csv')
sample_submission = pd.read_csv('/Path/to/your/submission.csv')
cos_sim = 1 - spatial.distance.cosine(sample_submission['val'].values, prompt_embeddings)
print(cos_sim)