# export OPENAI_API_KEY=YOUR_API_KEY
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the following commands to estimate the performance on emorynlp datasets

python emorynlp/chatgpt_ed.py --model_name chatgpt
# python emorynlp/chatgpt_si.py --model_name chatgpt
# python emorynlp/chatgpt_rs.py --model_name chatgpt
# python emorynlp/chatgpt_rg.py --model_name chatgpt

# python emorynlp/chatgpt_ed.py --model_name gpt-4
# python emorynlp/chatgpt_si.py --model_name gpt-4
# python emorynlp/chatgpt_rs.py --model_name gpt-4
# python emorynlp/chatgpt_rg.py --model_name gpt-4