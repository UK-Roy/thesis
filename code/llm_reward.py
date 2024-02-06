from localLLM import LocalLanguageModel
from prompt import system_prompts, prompt_templates, goal_strings, regexes, retry_prompts

llm = LocalLanguageModel(system_prompt=system_prompts[prompt_version],
                                      answer_regex=regexes[prompt_version],
                                      retry_prompt=retry_prompts[prompt_version],
                                      model_name=model_name, num_gpus=num_gpus,
                                      logdir=logdir)