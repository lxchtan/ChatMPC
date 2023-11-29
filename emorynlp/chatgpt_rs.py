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
import pandas as pd

def ensure(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path

logger = logging.getLogger(__file__)

def transfer_to_dialogue(fname):
  df = pd.read_csv(fname)
  dialogue = {}
  utt = df['Utterance'].tolist()
  dia_si = df['Scene_ID'].tolist()
  dia_se = df['Season'].tolist()
  dia_ep = df['Episode'].tolist()
  utt_id = df['Utterance_ID'].tolist()
  emotions = df['Emotion'].tolist()
  speakers = df['Speaker'].tolist()

  dia_id = [f"si{si}se{se}ep{ep}" for si, se, ep in zip(dia_si, dia_se, dia_ep)]

  for did, uid, u, e, s in zip(dia_id, utt_id, utt, emotions, speakers):
    if did in dialogue:
      assert uid - dialogue[did][-1]['uid'] >= 1
    else:
      dialogue[did] = []

    dialogue[did].append({
        'uid': uid,
        'utterance': u,
        'emotion': e.lower(),
        'speaker': eval(s)[0],
    })

  return dialogue

def load_dataset_rs(fname, n_negative=9):
  dialogues = transfer_to_dialogue(fname)
  # Check if pickle file exists.
  pickle_fname = fname.replace('.csv', f'_rs_{n_negative}neg.pkl')
  if os.path.exists(pickle_fname):
    with open(pickle_fname, 'rb') as pfile:
      dataset = pickle.load(pfile)
      print("Loaded dataset from pickle file.")
      return dataset

  ctx_list = []
  ctx_spk_list = []
  rsp_list = []
  rsp_spk_list = []

  for dialogue in dialogues.values():
    ctx = [line['utterance'] for line in dialogue[:-1]]
    ctx_spk = [line['speaker'] for line in dialogue[:-1]]
    rsp = dialogue[-1]['utterance']
    rsp_spk = dialogue[-1]['speaker']

    ctx_list.append(ctx)
    ctx_spk_list.append(ctx_spk)
    rsp_list.append(rsp)
    rsp_spk_list.append(rsp_spk)
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

def main(
    model_name: str,
    with_spk: bool = True,
):
  
  if model_name == 'chatgpt':
    model = "gpt-3.5-turbo-0301"
  elif model_name == 'gpt-4':
    model = "gpt-4-0314"

  single_turn_chat_model = partial(single_turn_chat, model=model)

  print(f"Using Model {model}!")
  test = "data/emorynlp/emorynlp_test_final.csv"

  prompter = Prompter("alpaca_short")
  
  instruction = """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to select the most appropriate response from the candidate set. The output format is "#{i} -- {speaker}: {utterance}".""" if with_spk else """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to select the most appropriate response from the candidate set. The output format is "#{i} -- {utterance}"."""
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_rs(test)
  data_nums = len(test_set)
  print(len(test_set))

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[1]
    dialogue_history_spk = data[2]
    history = [
      f"{dhs}: {dh}" if with_spk else dh for dh, dhs in zip(dialogue_history, dialogue_history_spk)
    ]
    rsp_candidates = data[4]
    rsp_spk = data[5]
    candidates = [
      f"#{i} -- {rsp_spk}: {dh}" if with_spk else f"#{i} -- {dh}" for i, dh in enumerate(rsp_candidates)
    ]
    example = '--\nDialogue History:\n' + '\n'.join(history) + '\n--\nCandidates:\n' + '\n'.join(candidates)
    label = data[-1][0]
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
  result_file = f"emorynlp/results/rs/{model_name}_{data_nums}s_withspk{with_spk}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  fire.Fire(main)
