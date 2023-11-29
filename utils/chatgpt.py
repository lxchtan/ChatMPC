import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import tiktoken


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def single_turn_chat(prompt, temperature=0, model="gpt-3.5-turbo-0301"):
    messages = [
        {"role": "user", "content": prompt},
    ]
    completion = completion_with_backoff(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    response = completion["choices"][0]["message"]["content"]
    return response


def cal_chatgpt_cost(prompts, model, task):
    pass
    # # model: "gpt-3.5-turbo-0301", "gpt-4-0314"
    # # task: "ed", "ar", "si", "rs", "rg"
    # cost_dict = {
    #     "gpt-3.5-turbo-0301": [0.0010, 0.0020],
    #     "gpt-4-0314": [0.03, 0.06],
    # }
    # tokens = 0
    # for prompt in prompts:
    #     messages = [
    #         {"role": "user", "content": prompt},
    #     ]
    #     tokens += num_tokens_from_messages(messages, model)

    # prompt_cost = cost_dict[model][0] * tokens / 1000
    # generation_cost = cost_dict[model][1] * tokens / 1000

    # print(f"Prompts cost: ${prompt_cost:.4f}.")
    # print(f"Estimate total cost: ${prompt_cost + generation_cost:.4f}.")
    # # print(f"Total may cost: ${0.002 * (tokens + 50 * len(prompts)) / 1000:.4f}.")
