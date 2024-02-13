import logging

from localLLM import LocalLanguageModel
from prompt import system_prompts, task_reward_signature_string, reward_signature

import numpy as np

logging.basicConfig(level=logging.INFO, filename="log/log2.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

llm = LocalLanguageModel(system_prompt=system_prompts,
                         signature = task_reward_signature_string
                        # answer_regex=regexes,
                        # retry_prompt=retry_prompts,
                        # model_name=model_name, 
                        # logdir=logdir
                        )

results = llm.generate(task_reward_signature_string)
result_output = [output.outputs[0].text for output in results]

logging.info(result_output)

output = ''.join(result_output)

reward_function = output[output.find('def') : output.find('}')+1]

with open("reward/reward1.txt", "w") as text_file:
    text_file.write(reward_function)