from localLLM import LocalLanguageModel
from prompt import system_prompts, task_reward_signature_string, reward_signature

import numpy as np

llm = LocalLanguageModel(system_prompt=system_prompts,
                         signature = reward_signature
                        # answer_regex=regexes,
                        # retry_prompt=retry_prompts,
                        # model_name=model_name, 
                        # logdir=logdir
                        )

results = llm.generate(task_reward_signature_string)