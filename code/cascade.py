import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')

probs_conv = np.load(os.path.join(DATA_PATH, 'probs_conv.npy'))
probs_maxvit = np.load(os.path.join(DATA_PATH, 'probs_maxvit.npy'))
probs_spec = np.load(os.path.join(DATA_PATH, 'probs_convspec.npy'))  #특공대 [7,3,4,14]

TARGET_CLASSES = [7, 3, 4, 14]
refined_probs_conv = probs_conv.copy()

for i in range(len(refined_probs_conv)):
    if refined_probs_conv[i].argmax() in TARGET_CLASSES:
        for spec_idx, cls_id in enumerate(TARGET_CLASSES):
            refined_probs_conv[i][cls_id] = (refined_probs_conv[i][cls_id] * 0.3) + (probs_spec[i][spec_idx] * 0.7)

probs_ensemble = np.zeros_like(probs_maxvit)

for i in range(17):
    if i in [7, 3, 4, 14]:
        probs_ensemble[:, i] = (probs_maxvit[:, i] * 0.2) + (refined_probs_conv[:, i] * 0.8)
    elif i == 6:
        probs_ensemble[:, i] = (probs_maxvit[:, i] * 0.7) + (refined_probs_conv[:, i] * 0.3)
    elif i in [11, 13]:
        probs_ensemble[:, i] = (probs_maxvit[:, i] * 0.6) + (refined_probs_conv[:, i] * 0.4)
    elif i == 12:
        probs_ensemble[:, i] = (probs_maxvit[:, i] * 0.4) + (refined_probs_conv[:, i] * 0.6)
    else:
        probs_ensemble[:, i] = (probs_maxvit[:, i] * 0.5) + (refined_probs_conv[:, i] * 0.5)

final_preds = probs_ensemble.argmax(axis=1)
test_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
out_df = pd.DataFrame({'ID': test_df['ID'], 'target': final_preds})

output_name = 'output.csv'
out_df.to_csv(os.path.join(DATA_PATH, output_name), index=False)
