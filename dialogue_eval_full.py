import sys
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

from utils import convert_to_json
from metric.evaluator import get_evaluator

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import importlib
ckpts = importlib.import_module("lit-gpt.scripts.batch_convert_lit_checkpoint")

chosen = ckpts.oasst1_dolly_lingua

# Example for dialogue response generation
task = 'dialogue'
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# dimensions: ['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']

# # a list of dialogue histories
# src_list = ['hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n']
# # a list of additional context that should be included into the generated response
# context_list = ['the 3 horizontal line menu on apps and websites is called a hamburger button .\n']
# # a list of model outputs to be evaluated
# output_list = ['i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?']

def row_eval(row):
    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=[row['output']], src_list=[row['input']], ref_list=[row['label']], context_list=[''])

    # Get multi-dimensional evaluation scores
    scores_dict = {}
    eval_scores = evaluator.evaluate(data, print_result=False)
    dims = list(eval_scores[0].keys())
    for dim in dims:
        cur_score = sum(score[dim] for score in eval_scores)
        scores_dict[dim] = round(cur_score / len(eval_scores), 6)
    return scores_dict

for model_name, _, checkpoint_name, dataset_name in chosen:
    name = '-'.join([model_name, dataset_name, checkpoint_name])
    print(name)
    df = pd.read_pickle(wd / f'lit-gpt/out/inference/{name}.pkl')
    scores = df.progress_apply(row_eval, axis=1, result_type='expand')
    print(scores.describe())
    final = pd.concat([df, scores], axis=1)
    final.to_pickle(wd / f'lit-gpt/out/inference/{name}-eval.pkl')
