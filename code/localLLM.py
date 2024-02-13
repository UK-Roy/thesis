import os

import numpy as np
import re
from typing import List, Optional, Sequence

from vllm import LLM, SamplingParams
from fastchat.model.model_adapter import get_conversation_template

class AnnotationIdx:
    FIRST = 0
    SECOND = 1
    TIE = 2
    UNKOWN = 3

class LocalLanguageModel:
    def __init__(
        self,
        system_prompt: str,
        signature = str,
        # answer_regex: str,
        # retry_prompt: str,
        model_name: str = 'meta-llama/Llama-2-7b-chat-hf',
        # num_gpus: int = 8,
        logdir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        # self.answer_regex = answer_regex
        # self.retry_prompt = retry_prompt
        self.llm = LLM(model=model_name, dtype='float16', 
                       max_num_batched_tokens=4096)
        self.system_prompt = system_prompt,
        self.reward_sig = signature
        self.logdir = logdir
        if self.logdir is not None:
            # Create directory
            os.makedirs(self.logdir, exist_ok=True)
    
    def generate(self, messages: List[str]) -> List[int]:
        prompts = f"{self.system_prompt} \n {self.reward_sig}"
        # convs = []
        
        # for message in messages:
        #     conv = get_conversation_template(self.model_name)
        #     conv.system = self.system_prompt
        #     conv.append_message(conv.roles[0], message)
        #     conv.append_message(conv.roles[1], None)
        #     prompt = conv.get_prompt()
        #     prompts.append(prompt)
        #     convs.append(conv)

        sampling_params = SamplingParams(top_k=50, max_tokens=4096,
                                         temperature=0.9, top_p=0.95,
                                         )
        outputs = self.llm.generate(prompts, sampling_params)

        return outputs