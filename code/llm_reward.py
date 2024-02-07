from localLLM import LocalLanguageModel
from prompt import system_prompts, task_reward_signature_string

import numpy as np

p = system_prompts
t = task_reward_signature_string

llm = LocalLanguageModel(system_prompt=system_prompts,
                        # answer_regex=regexes,
                        # retry_prompt=retry_prompts,
                        # model_name=model_name, 
                        # logdir=logdir
                        )

results = llm.generate()