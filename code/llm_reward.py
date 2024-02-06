from localLLM import LocalLanguageModel
from prompt import system_prompts, prompt_templates, goal_strings, regexes, retry_prompts

import numpy as np

llm = LocalLanguageModel(system_prompt=system_prompts,
                        # answer_regex=regexes,
                        # retry_prompt=retry_prompts,
                        # model_name=model_name, 
                        logdir=logdir)

results = llm.generate(prompt)