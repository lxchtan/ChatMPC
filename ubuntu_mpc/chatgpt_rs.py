import pickle

import logging
import os
import json
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
from functools import partial
from utils.eval import calculate_metrics

from tqdm import tqdm
from pathlib import Path
import time
import fire
import re
import random

def ensure(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path

logger = logging.getLogger(__file__)

def enhance_dataset(fname, n_negative=9, enh_name="enh"):
  pickle_fname = fname.replace('.json', f'_rs_{n_negative}neg.pkl')
  with open(pickle_fname, 'rb') as pfile:
    old_dataset = pickle.load(pfile)
    print("Loaded dataset from pickle file.")

  relation_at_list = []

  with open(fname, 'r') as f:
    for line in f:
      data = json.loads(line)
      ctx = data['context']
      ans_idx = data['ans_idx']
      relation = [[0, -1]] + data['relation_at'] + [[len(ctx), ans_idx]]
      relation = [r[1] for r in relation]
      relation_at_list.append(relation)
      
  print("matched context-response pairs: {}".format(len(relation_at_list)))

  dataset = []
  for i in range(len(relation_at_list)):
    dataset.append((*old_dataset[i], relation_at_list[i]))

  print("dataset_size: {}".format(len(dataset)))

  new_pickle_fname = pickle_fname.replace("neg.pkl", f"neg_{enh_name}.pkl")
  # Save the dataset as a pickle file
  with open(new_pickle_fname, 'wb') as pfile:
    pickle.dump(dataset, pfile)
    print(f"Saved dataset to pickle file {new_pickle_fname}")

  return dataset

def load_dataset_rs(fname, n_negative=9, enhance=None):
  if enhance is not None:
    dataset = enhance_dataset(fname, n_negative, enhance)
    pickle_fname = fname.replace('.json', f'_rs_{n_negative}neg_{enhance}.pkl')
  else:
    pickle_fname = fname.replace('.json', f'_rs_{n_negative}neg.pkl')

  # Check if pickle file exists.
  if os.path.exists(pickle_fname):
    with open(pickle_fname, 'rb') as pfile:
      dataset = pickle.load(pfile)
      print("Loaded dataset from pickle file.")
      return dataset

  ctx_list = []
  ctx_spk_list = []
  rsp_list = []
  rsp_spk_list = []

  with open(fname, 'r') as f:
    for line in f:
      data = json.loads(line)
      ctx_list.append(data['context'])
      ctx_spk_list.append(data['ctx_spk'])
      rsp_list.append(data['answer'])
      rsp_spk_list.append(data['ans_spk'])
  print("matched context-response pairs: {}".format(len(ctx_list)))

  dataset = []
  index_list = list(range(len(ctx_list)))
  for i in range(len(ctx_list)):
    ctx = ctx_list[i]
    ctx_spk = ctx_spk_list[i]

    # positive
    rsp = rsp_list[i]
    rsp_spk = rsp_spk_list[i]

    # negative
    negatives = random.sample(index_list, n_negative)
    while i in negatives:
      negatives = random.sample(index_list, n_negative)
    assert i not in negatives

    candidate_ids = [i] + negatives
    random.shuffle(candidate_ids)
    pos_i = candidate_ids.index(i)

    dataset.append((i, ctx, ctx_spk, candidate_ids, [rsp_list[cid] for cid in candidate_ids], rsp_spk, (pos_i, rsp)))

  print("dataset_size: {}".format(len(dataset)))

  # Save the dataset as a pickle file
  with open(pickle_fname, 'wb') as pfile:
    pickle.dump(dataset, pfile)
    print(f"Saved dataset to pickle file {pickle_fname}")

  return dataset

# 方法二
def intToRoman2(num: int) -> str:
    a = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    b = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    res = ''
    num += 1
    for i, n in enumerate(a):
        while num >= a[i]:
            res += b[i]
            num -= a[i]
    return res

def main(
    model_name: str,
    with_spk: bool = True,
    with_ar: bool = False,
):
  if model_name == 'chatgpt':
    model = "gpt-3.5-turbo-0301"
  elif model_name == 'gpt-4':
    model = "gpt-4-0314"

  single_turn_chat_model = partial(single_turn_chat, model=model)

  print(f"Using Model {model}!")

  test = "data/MPC/test.json"

  prompter = Prompter("alpaca_short")

  instruction = "You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to select the most appropriate response from the candidate set. "
  instruction += """The output format is "#{i} -- Speaker {speaker}: {utterance}".""" if with_spk else """The output format is "#{i} -- {utterance}"."""
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_rs(test, enhance="relation")
  data_nums = len(test_set)
  print(len(test_set))

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[1]
    dialogue_history_spk = data[2]
    relations = data[-1]
    history_relation = relations[:-1]
    response_relation = relations[-1]
    if not with_ar:
      history = [
        f"Speaker {dhs}: {dh}" if with_spk else dh for dh, dhs in zip(dialogue_history, dialogue_history_spk)
      ]
    else:
      history = [
        (f"[Reply to #U-{intToRoman2(idx)} -- Speaker {dialogue_history_spk[idx]}: {dialogue_history[idx]}] #U-{intToRoman2(i)} -- Speaker {dhs}: {dh} " if i > 0 else f"#U-{intToRoman2(i)} -- Speaker {dhs}: {dh}") if with_spk else (f"[Reply to #U-{intToRoman2(idx)} -- {dialogue_history[idx]}] #U-{intToRoman2(i)} -- {dh} " if i > 0 else f"#U-{intToRoman2(i)} -- {dh}") for i, (dh, dhs, idx) in enumerate(zip(dialogue_history, dialogue_history_spk, history_relation))
      ]
    gen_prompt = ''
    if with_ar:
      gen_prompt = f"\n--\nNote that the response is [Reply to #U-{intToRoman2(response_relation)} -- Speaker {dialogue_history_spk[response_relation]}: {dialogue_history[response_relation]}]" if with_spk else f"\n--\nNote that the response is [Reply to #U-{intToRoman2(response_relation)} -- {dialogue_history[response_relation]}]"

    rsp_candidates = data[4]
    rsp_spk = data[5]
    candidates = [
      f"#{i} -- Speaker {rsp_spk}: {dh}" if with_spk else f"#{i} -- {dh}" for i, dh in enumerate(rsp_candidates)
    ]
    example = '--\nDialogue History:\n' + '\n'.join(history) + gen_prompt + '\n--\nCandidates:\n' + '\n'.join(candidates)
    label = data[-2][0]
    prompt = prompter.generate_prompt(instruction, example)
    prompts.append(prompt)
    answers.append(label)

  cal_chatgpt_cost(prompts, model, task='rs')

  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]

  try:
    predictions2 = [int(re.findall(r'\d+', p.split('--')[0])[0]) for p in predictions]
    scores = calculate_metrics(predictions2, answers, deal_preds=False)
  except:
    scores = -1

  results = [{
    "instruction": instruction,
    "model": model,
    "prompt": prompter.generate_prompt(instruction, example),
    "scores": scores
  }]
  print(scores)

  for prediction, answer, prompt in zip(predictions, answers, prompts):
    results.append({
      "prompt": prompt,
      "prediction": prediction,
      "answer": answer
    })
  result_file = f"ubuntu_mpc/results/rs/{model_name}_{data_nums}s_withspk{with_spk}_withar{with_ar}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)
  
if __name__ == '__main__':
  fire.Fire(main)