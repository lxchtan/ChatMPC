import logging
import json
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
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

def load_dataset_ed(fname):
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
      'speaker': eval(s)[0],
      'utterance': u,
      'emotion': e.lower(),
    })
  
  dataset = [
    [
      did,
      [d['uid'] for d in data],
      [d['speaker'] for d in data[:-1]],
      [d['utterance'] for d in data[:-1]],
      [d['emotion'] for d in data],
      data[-1]['speaker'],
      data[-1]['utterance'],
    ]
      for did, data in dialogue.items()
      if data[-1]['speaker'] in [d['speaker'] for d in data[:-1]] and len(set([d['speaker'] for d in data[:-1]])) > 1
  ]

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
  test = "data/emorynlp/emorynlp_test_final.csv"

  prompter = Prompter("alpaca_short")
  
  instruction = "You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Please identify the speaker of the last sentence. The output format should be: #{num} -- {speaker}: {utterance}"
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_ed(test)
  data_nums = len(test_set)
  print(len(test_set))

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history_spk = data[2]
    dialogue_history = data[3]
    rsp = data[-1]
    history = [
      f"#{i} -- {dhs}: {dh}" for i, (dh, dhs) in enumerate(zip(dialogue_history, dialogue_history_spk))
    ]
    history += [
      f"#{len(dialogue_history)} -- {rsp}"
    ]
    example = '\n'.join(history)
    label = data[-2]
    prompt = prompter.generate_prompt(instruction + f'\nNote that the speaker is one of [{set(dialogue_history_spk)}].', example)
    prompts.append(prompt)
    answers.append(label)

  cal_chatgpt_cost(prompts)

  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]
  
  try:
    predictions2 = [p.split(':')[0].split('--')[1].strip() for p in predictions]
    scores = sum([(1 if p.lower()==a.lower() else 0) for p, a in zip(predictions2, answers)]) / len(predictions2)
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
  result_file = f"emorynlp/results/si/{model_name}_{data_nums}s_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  fire.Fire(main)
