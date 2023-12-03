import logging
import json
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
from functools import partial

from tqdm import tqdm
from pathlib import Path
import time
import fire
from utils.eval import calculate_generation_metrics
import re

def ensure(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


logger = logging.getLogger(__file__)


def load_dataset_rg(fname):
  dataset = []
  with open(fname, 'r') as f:
    for line in f:
      data = json.loads(line)
      ctx = data['context']
      ctx_spk = data['ctx_spk']
      ctx_adr = data['ctx_adr']
      rsp = data['answer']
      rsp_spk = data['ans_spk']
      rsp_adr = data['ans_adr']
      ans_idx = data['ans_idx']
      relation = [[0, -1]] + data['relation_at'] + [[len(ctx), ans_idx]]
      relation = [r[1] for r in relation]
      dataset.append((ctx, ctx_spk, rsp, rsp_spk, rsp_adr, relation))

  print("dataset_size: {}".format(len(dataset)))
  return dataset

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
  
  instruction = "You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to generate the most appropriate response. "
  if not with_ar:
    instruction += """The output format is "Speaker {rsp_spk}: {rsp}".""" if with_spk else ""
  else:
    instruction += """The output format is "[Reply to #{reply_uid} -- Speaker {reply_spk}: {reply_utterance}] Speaker {rsp_spk}: {rsp}".""" if with_spk else """The output format is "[Reply to #{reply_uid} -- {reply_utterance}] {rsp}"."""
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_rg(test)
  data_nums = len(test_set)
  print(data_nums)

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[0]
    dialogue_history_spk = data[1]
    relations = data[-1]
    history_relation = relations[:-1]
    response_relation = relations[-1]

    if with_ar:
      history = [
        (f"[Reply to #{idx} -- Speaker {dialogue_history_spk[idx]}: {dialogue_history[idx]}] #{i} -- Speaker {dhs}: {dh} " if i > 0 else f"#{i} -- Speaker {dhs}: {dh}") if with_spk else (f"[Reply to #{idx} -- {dialogue_history[idx]}] #{i} -- {dh} " if i > 0 else f"#{i} -- {dh}") for i, (dh, dhs, idx) in enumerate(zip(dialogue_history, dialogue_history_spk, history_relation))
      ] 
    else:
      history = [
        f"Speaker {dhs}: {dh}" if with_spk else dh for dh, dhs in zip(dialogue_history, dialogue_history_spk)
      ]

    rsp = data[2]
    rsp_spk = data[3]
    gen_prompt = ''
    if with_spk and not with_ar:
      gen_prompt = f'\n--\nPlease give a response on behalf of Speaker {rsp_spk}.\n'
    if with_spk and with_ar:
      gen_prompt = f'\n--\nPlease give a response on behalf of Speaker {rsp_spk} for Uttenrance #{response_relation}.\n'
      gen_prompt += f"The part of response is [Reply to #{response_relation} -- Speaker {dialogue_history_spk[response_relation]}: {dialogue_history[response_relation]}] Speaker {rsp_spk}: \n"
      gen_prompt += "Please finish the response generation."
    if not with_spk and with_ar:
      gen_prompt = f'\n--\nPlease give a response for Utterance #{response_relation}.\n'
      gen_prompt += f"The part of response is [Reply to #{response_relation} -- {dialogue_history[response_relation]}] \n"
      gen_prompt += "Please finish the response generation."


    example = '--\nDialogue History:\n' + '\n'.join(history) + gen_prompt
    prompt = prompter.generate_prompt(instruction, example)
    prompts.append(prompt)
    answers.append(rsp)

  cal_chatgpt_cost(prompts, model, task='rg')

  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]

  try:
    predictions2 = [re.sub(u"\\[Reply.*?]", "", s).strip() for s in predictions]
    predictions2 = [re.sub(u"Speaker \d: ", "", s).strip()  for s in predictions2]
    predictions2 = [re.sub(u"#\d -- ", "", s).strip()  for s in predictions2]
    predictions2 = [s[1:].strip() if s.startswith(':') else s for s in predictions2]
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
  result_file = f"ubuntu_mpc/results/rg/{model_name}_{data_nums}s_withspk{with_spk}_withar{with_ar}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)
  

if __name__ == '__main__':
  fire.Fire(main)