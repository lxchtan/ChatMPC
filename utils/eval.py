import evaluate
import re

def calculate_ed(preds, refs):
  f1_metric = evaluate.load("f1")
  results = f1_metric.compute(predictions=preds, references=refs, average='weighted')
  return results

def calculate_generation_metrics(preds, refs):
  generation_metric = evaluate.combine(["sacrebleu", "rouge", "meteor"], force_prefix=True)
  results = generation_metric.compute(predictions=preds, references=refs)

  results['max'] = "sacrebleu=100, rouge=1.0, meteor=1.0, mauve=1.0, bertscore=1.0"

  return results

def calculate_metrics(preds, refs, deal_preds=False):
  if deal_preds:
    preds = [int(re.findall(r'\d+', p)[0]) for p in preds]
  f1_metric = evaluate.load("accuracy")
  results = f1_metric.compute(predictions=preds, references=refs)
  return results
