import replicate
import os
import logging

# from utils.read_file import file_to_string
from read_file import file_to_string

logging.basicConfig(level=logging.INFO, filename="log/log.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["REPLICATE_API_TOKEN"] =  file_to_string("api/api.txt")
model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

code = file_to_string("prompt/code.txt")
reward_signature = file_to_string("prompt/reward_sig.txt")

output = replicate.run(
    model,
    input={
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": f"{code}",
        "temperature": 0.5,
        "system_prompt": "You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible. Your goal is to write a reward function for the environment that will help the agent learn the task described in text. Your reward function should use useful variables from the environment as inputs.  As an example, the reward function signature can be: {reward_signature}",
        "max_new_tokens": 500,
        "min_new_tokens": -1
    }
)

full_response = ""

for item in output:
    full_response += item

logging.info(full_response)
