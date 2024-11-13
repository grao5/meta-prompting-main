# Import the necessary libraries
import random
import re
import time
import retry

# Import the typing library for type hints
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the necessary classes from the utils folder
from .language_model import OpenAI_LanguageModel, xlam_LanguageModel
from .execute_code import execute_code_with_timeout

# Import the necessary dependencies at the top of the file
from transformers import StoppingCriteria


'''
s
Useful code: https://github.com/SalesforceAIResearch/xLAM/blob/dee27c4e3f47eea8266d0e504081ed07ed239181/xLAM/train/fm_datasets/base.py#L40
 elif "mixtral" in tokenizer.name_or_path.lower():
                self.BOT = tokenizer.bos_token if not tokenizer.add_bos_token else ""
                self.EOP = "[/INST]"
                self.EOT = tokenizer.eos_token if not tokenizer.add_eos_token else ""
                # self._response_template = f"<|assistant|>\n"
                self._response_template = self.EOP
                self._instruction_template = "[INST]"


might have to replace stop_token with implementation of stopping criteria for hugging face models.

Code from gpt:# 
# Example: Create a stopping criteria class if needed
from transformers import StoppingCriteria

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return any(token in input_ids[-1] for token in self.stop_tokens)

stopping_criteria = CustomStoppingCriteria(stop_tokens)

meta_model_output = self.language_model.generate(
    prompt_or_messages=formatted_prompt,
    max_new_tokens=max_tokens,
    stopping_criteria=[stopping_criteria],
    num_return_sequences=num_return_sequences,
    temperature=temperature,
    top_p=top_p,
    **kwargs,
)


'''

# Define the MetaPromptingScaffolding class
class MetaPromptingScaffolding:
    def __init__(
        self,
        language_model: Union[OpenAI_LanguageModel, xlam_LanguageModel],
        generator_settings: Dict[str, Any],
        verifier_settings: Dict[str, Any],
        summarizer_settings: Dict[str, Any],
        error_message: str,
        final_answer_indicator: str,
        expert_python_message: str,
        intermediate_feedback: str,
        fresh_eyes: bool = False,
        include_expert_name_in_instruction: bool = False,
        extract_output: bool = False,
        use_zero_shot_cot_in_expert_messages: bool = False,
    ) -> None:
        # Set the language model
        self.language_model = language_model

        # Set the generator and verifier parameters + summarizer parameters (optional)
        self.generator_settings = generator_settings
        self.verifier_settings = verifier_settings
        self.summarizer_settings = summarizer_settings

        # Set the error message and final answer indicator
        self.error_message = error_message
        self.final_answer_indicator = final_answer_indicator

        # Set the fresh_eyes flag
        self.fresh_eyes = fresh_eyes

        # Other helper variables and constants for the model
        self.triple_quotes = '"""'

        # Set the include_expert_name_in_instruction flag
        self.include_expert_name_in_instruction = include_expert_name_in_instruction
        self.extract_output = extract_output
        self.expert_python_message = expert_python_message
        self.intermediate_feedback = intermediate_feedback
        self.use_zero_shot_cot_in_expert_messages = use_zero_shot_cot_in_expert_messages

    @retry.retry(tries=7, delay=5)
    def meta_model_generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 1024,
        num_return_sequences: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.95,
        counter: int = 0,
        last_answer: str = None,
        original_question: str = None,
        trial_num: int = 0,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        try:
            assert self.language_model.model_type == "chat"
            assert num_return_sequences == 1

            # This step is defined to ensure that the meta model returns a response in less than 16 rounds.
            # Note: Please feel free to change the number of rounds as you see fit.
            if counter == 16:
                return prompt_or_messages

            # TODO(msuzgun)[improvement]: In the future, if the total content is to long, we can use the summarizer to summarize the whole content.

            entire_message_log = prompt_or_messages.copy()

            while True:
                entire_message_log[-1][
                    "content"
                ] = f"ROUND {counter+1}:\n\n{entire_message_log[-1]['content']}"

                if counter == 14:
                    entire_message_log[-1][
                        "content"
                    ] += (
                        f"This is the last round; so, please present your final answer."
                    )

                # Step 1: Generate an output from the meta model
                # potential changes for xlam (sft mistral)
                # max_tokens -> max_new_tokens
                # 
                meta_model_output = self.language_model.generate(
                    prompt_or_messages=entire_message_log,
                    stop_tokens=stop_tokens,
                    max_tokens=max_tokens,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )[0]

                # TODO(msuzgun)[improvement]: If the generated output is too long, then we can use the summarizer to summarize the content.

                entire_message_log.append(
                    {"role": "assistant", "content": meta_model_output}
                )

                # Check if the meta_model_output contains a text of the form "Expert XYZ:\n" (where XYZ is an alphabanumeric string).

                # Step 2 (a): If we are not in the 0-shot CoT setting, check if the meta model output contains any text between triple quotes.
                # If it does, then generate an output from the corresponding model.
                pattern = r"Expert ((?:\w+ ?){1,5}):\n"
                if (self.fresh_eyes) and (
                    # f":\n{self.triple_quotes}" in meta_model_output
                    re.search(pattern, meta_model_output)
                ):
                    # There might be multiple instructions between the triple quotes; so, split the output by the triple quotes.
                    triple_quote_splits = meta_model_output.split(self.triple_quotes)
                    # Odd indices are the instructions, even indices contain the lines preceding the instructions (indicating which model to use).
                    len_triple_quote_splits = len(triple_quote_splits)

                    intermediate_output = ""
                    model_num_return_sequences = 1  # Feel free to ignore the model_num_return_sequences > 1 case for now.
                    # Iterate over the instructions.
                    for i in range(1, len_triple_quote_splits, 2):
                        # Get the instructions for the corresponding model, as well as the line preceding the instructions (indicating which Expert to use).
                        line_preceding_instruction = triple_quote_splits[i - 1].strip()
                        model_name = line_preceding_instruction.split("\n")[-1].strip()
                        if "Expert " in model_name:
                            if model_name[-1] == ":":
                                model_name = model_name[:-1]

                            model_instruction = triple_quote_splits[i].strip()

                            # Add the expert name to the instruction.
                            if self.include_expert_name_in_instruction:
                                model_instruction = (
                                    f"You are {model_name}.\n\n{model_instruction}"
                                )

                            # Add "Let's think step by step." to the instruction.
                            if self.use_zero_shot_cot_in_expert_messages:
                                model_instruction += f"\n\nLet's think step by step."

                            # By default, we use the generator Expert to generate an output from the instructions.
                            model_temp = self.generator_settings["parameters"][
                                "temperature"
                            ]
                            model_top_p = self.generator_settings["parameters"]["top_p"]
                            model_max_tokens = self.generator_settings["parameters"][
                                "max_tokens"
                            ]
                            model_num_return_sequences = self.generator_settings[
                                "parameters"
                            ]["num_return_sequences"]

                            model_message_list = self.generator_settings["message-list"]
                            model_stop_tokens = None

                            current_model_message_list = model_message_list.copy()
                            current_model_message_list.append(
                                {
                                    "role": "user",
                                    "content": model_instruction,
                                }
                            )

                            if model_name == "Expert Python":
                                current_model_message_list[-1][
                                    "content"
                                ] = f"{self.expert_python_message}.\n\n{current_model_message_list[-1]['content']}"

                            # Finally, read to call the corresponding model.
                            model_outputs = self.generate(
                                prompt_or_messages=current_model_message_list,
                                max_new_tokens=model_max_tokens,
                                temperature=model_temp,
                                top_p=model_top_p,
                                num_return_sequences=model_num_return_sequences,
                                **kwargs
                            )

                            for _, model_output in enumerate(model_outputs):
                                ## Special case for Expert Python
                                if model_name == "Expert Python":
                                    # If the output contains the special substring, then we need to execute the code.
                                    if "Please run this code!" in model_output:
                                        # Get the code #ADDED: 14-08-2023
                                        code_text = model_output.split(
                                            "Please run this code!"
                                        )[0].strip()
                                        # Get the output
                                        code_text = code_text.replace(
                                            "```python", "```"
                                        )
                                        try:
                                            code_text = code_text.split("```")[
                                                -2
                                            ].strip()
                                        except:
                                            code_text = code_text.split("```")[
                                                1
                                            ].strip()

                                        # print(f"We are going to execute the following code:\n{code_text}\n\n")
                                        code_text = rf"{code_text}"
                                        # Execute the code
                                        python_output = execute_code_with_timeout(
                                            code_text
                                        )
                                        # Add the output to the model output
                                        model_output += f"Here is the Python code used to solve the problem:\n\n{code_text}\n\nHere is the output of the code when executed:\n\n{python_output}"

                                else:
                                    specicial_token = "* * *"
                                    if self.extract_output:
                                        # FIXME: Temporary fix
                                        if specicial_token in model_output:
                                            model_output = model_output.split(
                                                specicial_token
                                            )[1].strip()

                                        if len(model_output.split(" ")) > 128:
                                            model_output = (
                                                "Solution too long. Please try again."
                                            )
                                    else:
                                        model_output.replace(specicial_token, "")
                                        model_output.replace(
                                            "FINAL ANSWER:",
                                            f"{model_name}'s final answer:\n",
                                        )

                                intermediate_output += f"{model_name}'s output:\n{self.triple_quotes}\n{model_output}\n{self.triple_quotes}"

                            # Remove the last two newlines.
                            intermediate_output = intermediate_output.strip()

                            # TODO(msuzgun)[improvement]: Using an additional verifier and/or summarizer might be useful here.
                            # Feel free to ignore the following steps for now (for when model_num_return_sequences > 1).
                            if model_num_return_sequences > 1:
                                summarizer_prompt_or_messages = (
                                    self.summarizer_settings["message-list"].copy()
                                )
                                summarizer_prompt_or_messages.append(
                                    {
                                        "role": "user",
                                        "content": f"Please provide a clear and concise summary of the expert outputs, emphasizing the key similarities and differences between them.\n\nPrompt: {model_instruction}\n\nOutput: {intermediate_output}",
                                    }
                                )

                                # Let's call the summarizer Expert to summarize the outputs.
                                summarizer_output = self.language_model.generate(
                                    prompt_or_messages=summarizer_prompt_or_messages,
                                    stop_tokens=None,  # FIXME(msuzgun)
                                    max_tokens=self.summarizer_settings["parameters"][
                                        "max_tokens"
                                    ],
                                    num_return_sequences=self.summarizer_settings[
                                        "parameters"
                                    ]["num_return_sequences"],
                                    temperature=self.summarizer_settings["parameters"][
                                        "temperature"
                                    ],
                                    top_p=self.summarizer_settings["parameters"][
                                        "top_p"
                                    ],
                                    **kwargs,
                                )[0]

                                # Make this the new intermediate output.
                                intermediate_output = f"Here is the summary of {model_name}'s outputs:\n\n{summarizer_output}"

                    # Add the intermediate output to the full prompt or messages.
                    intermediate_output = (
                        f"{intermediate_output}\n\n{self.intermediate_feedback}"
                    )

                    # Add the intermediate output to the full prompt or messages.
                    entire_message_log.append(
                        {
                            "role": "user",
                            "content": intermediate_output,
                        }
                    )

                    # Prepare the prompt for the meta model
                    return self.meta_model_generate(
                        prompt_or_messages=entire_message_log,
                        stop_tokens=stop_tokens,
                        max_tokens=max_tokens,
                        num_return_sequences=num_return_sequences,
                        temperature=temperature,
                        top_p=top_p,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )
                # Step 2(b): Check if the meta_model_output contains the final answer indicator.
                elif self.final_answer_indicator in meta_model_output:
                    # The following code is commented out because we are not using the final answer indicator anymore.
                    # However, it is useful for debugging purposes.
                    # final_answer = meta_model_output.split(self.final_answer_indicator)[
                    #     -1
                    # ].strip()
                    # print(f"Final answer: {final_answer}")
                    return entire_message_log
                # Step 2(c): We need to continue the (meta-)conversation.
                else:
                    entire_message_log.append(
                        {"role": "user", "content": self.error_message}
                    )
                    
                    return self.meta_model_generate(
                        prompt_or_messages=entire_message_log,
                        stop_tokens=stop_tokens,
                        max_tokens=max_tokens,
                        num_return_sequences=num_return_sequences,
                        temperature=temperature,
                        top_p=top_p,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )

        except Exception as e:
            print(f"Houston, we have a problem in meta_model_generate: {e}")

            # If we have tried 7 times, then let's return the current prompt or messages.
            if trial_num == 7:
                return prompt_or_messages

            print("Waiting for 1-10 seconds...")
            # Let's wait for 1-10 seconds before trying again.
            time.sleep(random.randint(1, 10))
            return self.meta_model_generate(
                prompt_or_messages=entire_message_log,
                stop_tokens=stop_tokens,
                max_tokens=max_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                counter=counter,
                last_answer=last_answer,
                trial_num=trial_num + 1,
                **kwargs,
            )
    
    # ending of meta_model_generate function ^^

    @retry.retry(tries=7, delay=5)
    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 1024,
        num_return_sequences: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        model_output = self.language_model.generate(
            prompt_or_messages=prompt_or_messages,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )[0]

        return model_output

class xLAMMetaPromptingScaffolding:
    '''
    Mostly the same as metprompting, but for xLAM i just added the hugging face specifc params or reassigned the custom ones to huggingface
    '''
    def __init__(
        self,
        language_model: xlam_LanguageModel,
        generator_settings: Dict[str, Any],
        verifier_settings: Dict[str, Any],
        summarizer_settings: Dict[str, Any],
        error_message: str,
        final_answer_indicator: str,
        expert_python_message: str,
        intermediate_feedback: str,
        fresh_eyes: bool = False,
        include_expert_name_in_instruction: bool = False,
        extract_output: bool = False,
        use_zero_shot_cot_in_expert_messages: bool = False,
    ):
        self.language_model = language_model
        
        # Settings from original class
        self.generator_settings = generator_settings or {}
        self.verifier_settings = verifier_settings or {}
        self.summarizer_settings = summarizer_settings or {}
        self.error_message = error_message
        self.final_answer_indicator = final_answer_indicator
        self.fresh_eyes = fresh_eyes
        self.include_expert_name_in_instruction = include_expert_name_in_instruction
        self.extract_output = extract_output
        self.expert_python_message = expert_python_message
        self.intermediate_feedback = intermediate_feedback
        self.use_zero_shot_cot_in_expert_messages = use_zero_shot_cot_in_expert_messages
        
        # xLAM specific tokens
        self.INST = "[INST]"
        self.END_INST = "[/INST]"
        self.triple_quotes = '"""'
        
    def format_prompt(self, messages):
        # Convert messages to xLAM format
        formatted_text = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted_text += f"{self.INST}{msg['content']}{self.END_INST}"
            elif msg["role"] == "assistant":
                formatted_text += msg["content"]
        return formatted_text
    
    # def generate(
    #     self,
    #     prompt_or_messages,
    #     stop_tokens=None,  # Added to match original interface
    #     max_new_tokens=512,
    #     temperature=0.1,
    #     top_p=0.95,
    #     num_return_sequences=1,
    #     **kwargs
    # ):
    #     # Format the prompt
    #     if isinstance(prompt_or_messages, list):
    #         formatted_prompt = self.format_prompt(prompt_or_messages)
    #     else:
    #         formatted_prompt = f"{self.INST}{prompt_or_messages}{self.END_INST}"
            
    #     # Tokenize
    #     inputs = self.tokenizer(
    #         formatted_prompt,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True
    #     ).to(self.model.device)
        
    #     # Handle stop tokens if provided
    #     stopping_criteria = []
    #     if stop_tokens:
    #         from transformers import StoppingCriteria
    #         class StopOnTokens(StoppingCriteria):
    #             def __init__(self, stop_token_ids):
    #                 self.stop_token_ids = stop_token_ids
                
    #             def __call__(self, input_ids, scores):
    #                 for stop_id in self.stop_token_ids:
    #                     if stop_id in input_ids[0][-1:]:
    #                         return True
    #                 return False
                    
    #         stop_token_ids = [self.tokenizer.encode(token)[-1] for token in stop_tokens]
    #         stopping_criteria.append(StopOnTokens(stop_token_ids))
        
    #     # Generate
    #     outputs = self.model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         max_new_tokens=max_new_tokens,
    #         do_sample=(temperature > 0),
    #         temperature=temperature,
    #         top_p=top_p,
    #         num_return_sequences=num_return_sequences,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         stopping_criteria=stopping_criteria if stopping_criteria else None,
    #         **kwargs
    #     )
        
    #     # Decode
    #     generated_texts = []
    #     for output in outputs:
    #         decoded = self.tokenizer.decode(
    #             output[len(inputs["input_ids"][0]):],
    #             skip_special_tokens=True
    #         )
    #         generated_texts.append(decoded)
            
    #     return generated_texts[0] if num_return_sequences == 1 else generated_texts

    def meta_model_generate(
        self,
        inputs,
        stop_tokens=None,
        max_new_tokens=1024,
        num_return_sequences=1,
        temperature=0.1,
        top_p=0.95,
        counter=0,
        last_answer=None,
        original_question=None,
        trial_num=0,
        **kwargs
    ):
        """
        Implements the meta-model generation logic similar to the original class
        but using xLAM's generation mechanism
        """
        try:
            if counter == 16:
                return inputs

            entire_message_log = inputs.copy()
            
            while True:
                entire_message_log[-1]["content"] = f"ROUND {counter+1}:\n\n{entire_message_log[-1]['content']}"
                
                if counter == 14:
                    entire_message_log[-1]["content"] += "\nThis is the last round; so, please present your final answer."
                
                # potential changes: stop_tokens should be removed, prompt_
                meta_model_output = self.language_model.generate(
                    inputs=entire_message_log,
                    stop_tokens=stop_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    **kwargs
                )

                # Rest of the meta_model_generate logic remains similar
                # but adapted for xLAM's generation mechanism
                # ... (implement the rest of the logic following the original class)
                
                # This is a simplified version - you'll want to implement the full logic
                # including expert handling, fresh eyes, etc.
                
                if self.final_answer_indicator in meta_model_output:
                    return entire_message_log
                    
                entire_message_log.append({"role": "assistant", "content": meta_model_output})
                entire_message_log.append({"role": "user", "content": self.error_message})
                
                return self.meta_model_generate(
                    inputs=entire_message_log,
                    stop_tokens=stop_tokens,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    counter=counter + 1,
                    last_answer=last_answer,
                    original_question=original_question,
                    **kwargs
                )

        except Exception as e:
            print(f"Error in meta_model_generate: {e}")
            if trial_num >= 7:
                return inputs
                
            time.sleep(random.randint(1, 10))
            return self.meta_model_generate(
                inputs=inputs,
                stop_tokens=stop_tokens,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                counter=counter,
                last_answer=last_answer,
                trial_num=trial_num + 1,
                **kwargs
            )
