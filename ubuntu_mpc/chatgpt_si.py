import logging
import json
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
from functools import partial

from tqdm import tqdm
from pathlib import Path
import time
import fire
import re

def ensure(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path

logger = logging.getLogger(__file__)

def load_dataset_si(fname):
  dataset = []
  with open(fname, 'r') as f:
    for line in f:
      data = json.loads(line)
      ctx = data['context']
      ctx_spk = data['ctx_spk']
      rsp = data['answer']
      rsp_spk = data['ans_spk']
      ans_idx = data['ans_idx']
      relation = [[0, -1]] + data['relation_at'] + [[len(ctx), ans_idx]]
      relation = [r[1] for r in relation]
      assert len(ctx) == len(ctx_spk)

      utrs_same_spk_with_rsp_spk = []
      for utr_id, utr_spk in enumerate(ctx_spk):
        if utr_spk == rsp_spk:
          utrs_same_spk_with_rsp_spk.append(utr_id)

      if len(utrs_same_spk_with_rsp_spk) == 0:
        continue

      label = [0 for _ in range(len(ctx))]
      for utr_id in utrs_same_spk_with_rsp_spk:
        label[utr_id] = 1

      dataset.append((ctx, ctx_spk, rsp, rsp_spk, label, relation))

  print("dataset_size: {}".format(len(dataset)))
  return dataset


def main(
    model_name: str,
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

  instruction = "You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Please identify the speaker of the last sentence. The output format should be only one speaker."
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_si(test)
  data_nums = len(test_set)
  print(len(test_set))

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[0]
    dialogue_history_spk = data[1]
    response = data[2]
    if with_ar:
      relations = data[-1]
      history_relation = relations[:-1]
      response_relation = relations[-1]
      history = [
        f"[Reply to #{idx} -- Speaker {dialogue_history_spk[idx]}: {dialogue_history[idx]}] #{i} -- Speaker {dhs}: {dh} " if i > 0 else f"#{i} -- Speaker {dhs}: {dh}" for i, (dh, dhs, idx) in enumerate(zip(dialogue_history, dialogue_history_spk, history_relation))
      ] + [f"[Reply to #{response_relation} -- Speaker {dialogue_history_spk[response_relation]}: {dialogue_history[response_relation]}] #{len(dialogue_history)} -- {response}"]
    else:
      history = [
        f"#{i} -- Speaker {dhs}: {dh}" for i, (dh, dhs) in enumerate(zip(dialogue_history, dialogue_history_spk))
      ]
      history += [
        f"#{len(dialogue_history)} -- {response}"
      ]
    example = '\n'.join(history)
    label = data[3]
    spk_list = [f'Speaker {spk}' for spk in set(dialogue_history_spk)] 
    prompt = prompter.generate_prompt(instruction + f'\nNote that the speaker is one of [{spk_list}].', example)
    prompts.append(prompt)
    answers.append(label)

  cal_chatgpt_cost(prompts, model, task='si')

  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]
  
  try:
    predictions2 = []
    for p in predictions:
      pp = p.replace("Speaker", "").strip() 
      pp_nums = re.findall(r'\d+', pp)
      predictions2.append(int(pp_nums[0]) if len(pp_nums) > 0 else -1)
    scores = sum([(1 if p==a else 0) for p, a in zip(predictions2, answers)]) / len(predictions2)
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
  result_file = f"ubuntu_mpc/results/si/{model}_{data_nums}s_withar{with_ar}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  fire.Fire(main)
