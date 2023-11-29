import logging
import os
import json
import openai
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
from utils.eval import calculate_generation_metrics
from functools import partial

from tqdm import tqdm
from pathlib import Path
import time
import fire
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
  dia_id = df['Dialogue_ID'].tolist()
  utt_id = df['Utterance_ID'].tolist()
  emotions = df['Emotion'].tolist()
  speakers = df['Speaker'].tolist()

  for did, uid, u, e, s in zip(dia_id, utt_id, utt, emotions, speakers):
    if did in dialogue:
      assert uid - dialogue[did][-1]['uid'] >= 1
    else:
      dialogue[did] = []

    dialogue[did].append({
        'uid': uid,
        'utterance': u,
        'emotion': e,
        'speaker': s,
    })

  return dialogue


def load_dataset_rg(fname):
  dataset = []
  dialogues = transfer_to_dialogue(fname)
  for dialogue in dialogues.values():
    if len(dialogue) <= 1: continue
    ctx = [line['utterance'] for line in dialogue[:-1]]
    ctx_spk = [line['speaker'] for line in dialogue[:-1]]
    rsp = dialogue[-1]['utterance']
    rsp_spk = dialogue[-1]['speaker']

    integrate_ctx = ctx + [rsp]
    integrate_ctx_spk = ctx_spk + [rsp_spk]
    assert len(integrate_ctx) == len(integrate_ctx_spk)

    dataset.append((ctx, ctx_spk, rsp, rsp_spk))

  print("dataset_size: {}".format(len(dataset)))
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
  test = "data/MELD/test_sent_emo.csv"

  prompter = Prompter("alpaca_short")
  
  instruction = """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to generate the most appropriate response. The output format is "{rsp_spk}: {rsp}".""" if with_spk else """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to generate the most appropriate response."""
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_rg(test)
  data_nums = len(test_set)
  print(len(test_set))

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[0]
    dialogue_history_spk = data[1]
    history = [
      f"{dhs}: {dh}" if with_spk else dh for dh, dhs in zip(dialogue_history, dialogue_history_spk)
    ]
    rsp = data[2]
    rsp_spk = data[3]
    example = '--\nDialogue History:\n' + '\n'.join(history) + (f'\n--\nPlease give a response on behalf of {rsp_spk}.\n' if with_spk else '')
    prompt = prompter.generate_prompt(instruction, example)
    prompts.append(prompt)
    answers.append(rsp)

  cal_chatgpt_cost(prompts, model, task='rg')
  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]

  try:
    predictions2 = [':'.join(p.strip().split('\n')[0].split(':')[1:]).strip() if with_spk else p.strip().split('\n')[0] for p in predictions]
    scores = calculate_generation_metrics(predictions2, answers)
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
  result_file = f"meld/results/rg/{model_name}_{data_nums}s_withspk{with_spk}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  fire.Fire(main)
