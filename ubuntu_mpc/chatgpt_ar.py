import logging
import json
from utils.chatgpt import single_turn_chat, cal_chatgpt_cost
from utils.prompter import Prompter
from functools import partial


from tqdm import tqdm
from pathlib import Path
import time
import fire
from utils.eval import calculate_metrics
import re


def ensure(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path

logger = logging.getLogger(__file__)

def merge_contexts(file_name):
  result = []

  with open(file_name, 'r') as file:
    for line in file:
      data = json.loads(line)
      contexts = data['context']
      speakers = data['ctx_spk']

      merged_contexts = []
      current_speaker = speakers[0]
      current_context = contexts[0]

      for context, speaker in zip(contexts[1:], speakers[1:]):
        if speaker == current_speaker:
          current_context += ' ' + context
        else:
          merged_contexts.append(current_context)
          current_context = context
          current_speaker = speaker

      merged_contexts.append(current_context)
      result.append({**data, 'context': merged_contexts})

  return result


def load_dataset_ar(fname):
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

      integrate_ctx = ctx + [rsp]
      integrate_ctx_spk = ctx_spk + [rsp_spk]
      integrate_ctx_adr = ctx_adr + [rsp_adr]
      assert len(integrate_ctx) == len(integrate_ctx_spk)
      assert len(integrate_ctx) == len(integrate_ctx_adr)

      label = []
      for utr_id_adr, utr_adr in enumerate(integrate_ctx_adr):

        label_utr = [0 for _ in range(len(integrate_ctx))]
        for cand_utr_id_spk, cand_utr_spk in enumerate(
                integrate_ctx_spk[:utr_id_adr]):  # consider only the preceding utterances
          if cand_utr_spk == utr_adr:
            label_utr[cand_utr_id_spk] = 1
        label.append(label_utr)

      adr_label = []
      for l in label:
        if sum(l) == 0:
          adr_label.append(-1)
        else:
          for i, v in enumerate(l):
            if v == 1:
              adr_label.append(ctx_spk[i])
              break

      dataset.append((ctx, ctx_spk, rsp, rsp_spk, adr_label))

  print("dataset_size: {}".format(len(dataset)))
  return dataset


def load_dataset_si(fname):
  dataset = []
  with open(fname, 'r') as f:
    for line in f:
      data = json.loads(line)
      ctx = data['context']
      ctx_spk = data['ctx_spk']
      rsp = data['answer']
      rsp_spk = data['ans_spk']
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

      dataset.append((ctx, ctx_spk, rsp, rsp_spk, label))

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

  test = "data/MPC/test.json"

  prompter = Prompter("alpaca_short")

  instruction = """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to find the addressee of each utterance. The output format is "#{i} -- Speaker {speaker}: {utterance} // Reply to Speaker {addressee}".""" if with_spk else """You have been presented with a sequence of multi-party conversational turns, organized in chronological order. Your task is to find the reply-to utterance of each utterance. The output format is "#{i} -- {utterance} // Reply to #{reply_i}"."""

  instruction += "\nPlease start from #1 since #0 is the first utterance that has no reply-to utterance. You should not leave any utterance unattended."
  instruction += "\nNote that each utterance strictly replies to one of previous."
  instruction += "\nUse temperature=0, minimize unnecessary words to not get confused."

  test_set = load_dataset_ar(test)
  data_nums = len(test_set)
  print(data_nums)

  prompts = []
  answers = []
  for data in tqdm(test_set, desc="Generating prompts"):
    dialogue_history = data[0]
    dialogue_history_spk = data[1]
    response = data[2]
    response_spk = data[3]
    label = data[4]
    history = [
      f"#{i} -- Speaker {dhs}: {dh}" if with_spk else f"#{i} -- {dh}" for i, (dh, dhs) in enumerate(zip(dialogue_history, dialogue_history_spk))
    ] + [
      f"#{len(dialogue_history)} -- Speaker {response_spk}: {response}" if with_spk else f"#{len(dialogue_history)} -- {response}"
    ]
    example = '\n'.join(history)
    prompt = prompter.generate_prompt(instruction, example)
    prompts.append(prompt)
    answers.append(label)

  cal_chatgpt_cost(prompts, model, task='ar')
  predictions = [single_turn_chat_model(prompt) for prompt in tqdm(prompts, desc="ChatGPT")]

  try:
    predictions2 = []
    for pred, data in zip(predictions, test_set):
      dialogue_history_spk = data[1]
      response_spk = data[3]
      dialogue_history_spk.append(response_spk)
      pred = pred.replace('\n//', ' //')
      for p in pred.split('\n'):
        try:
          rep = re.findall(r'\d+', p.split('//')[1])
        except IndexError:
          rep = re.findall(r'\d+', p.split(':')[1])
        if len(rep) > 0:
          try:
            predictions2.append(int(rep[0]) if with_spk else dialogue_history_spk[int(rep[0])])
          except IndexError:
            predictions2.append(-1)
        else:
          predictions2.append(-1)
    answers2 = sum([label[1:] for label in answers], [])
    scores = calculate_metrics(predictions2, answers2, deal_preds=False)
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
  result_file = f"ubuntu_mpc/results/ar/chatgpt_{data_nums}s_withspk{with_spk}_{int(time.time())}.json"
  with open(ensure(result_file), "w") as f:
    json.dump(results, f, indent=2)

if __name__ == '__main__':
  fire.Fire(main)
