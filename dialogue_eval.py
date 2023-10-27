import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from utils import convert_to_json
from metric.evaluator import get_evaluator

model_name = "pythia-2.8b-deduped"
run_name = "driven-wood-164"
checkpoint_name = "iter-000299-ckpt"
dataset_name = "oasst1"

# Example for dialogue response generation
task = 'dialogue'
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)

# # a list of dialogue histories
# src_list = ['hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n']
# # a list of additional context that should be included into the generated response
# context_list = ['the 3 horizontal line menu on apps and websites is called a hamburger button .\n']
# # a list of model outputs to be evaluated
# output_list = ['i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?']

def row_eval(row):
    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=[row['out']], src_list=[row['label']], context_list=[])

    # Get multi-dimensional evaluation scores
    scores_dict = {}
    eval_scores = evaluator.evaluate(data, print_result=False)
    dims = list(eval_scores[0].keys())
    for dim in dims:
        cur_score = sum(score[dim] for score in eval_scores)
        scores_dict[dim] = round(cur_score / len(eval_scores), 6)
    return scores_dict

df = pd.read_csv(f'/data/echen/lit-gpt/out/inference/{model_name}-{dataset_name}-{checkpoint_name}.csv', names=['src', 'out', 'label'])
scores = df.progress_apply(row_eval, axis=1, result_type='expand')
final = pd.concat(df, scores, axis=1)
final.to_csv(f'/data/echen/lit-gpt/out/inference/{model_name}-{dataset_name}-{checkpoint_name}-eval.csv', index=False)
