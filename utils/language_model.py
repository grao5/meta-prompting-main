# Import the necessary libraries
import os
import openai
from openai import OpenAI
from typing import Any, Dict, List, Optional, Union
import retry


class LanguageModel:
    """Abstract class for a language model."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.stop_tokens = stop_tokens or []
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        raise NotImplementedError("generate() not implemented.")

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_name

    def get_stop_tokens(self) -> List[str]:
        """Get the stop tokens."""
        return self.stop_tokens

    def get_kwargs(self) -> Any:
        """Get the kwargs."""
        return self.kwargs

    def set_kwargs(self, kwargs: Any) -> None:
        """Set the kwargs."""
        self.kwargs = kwargs


class DummyLanguageModel(LanguageModel):
    """A dummy language model that just returns the prompt."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, stop_tokens, **kwargs)

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        return [prompt] * num_return_sequences


class OpenAI_LanguageModel(LanguageModel):
    """A language model from OpenAI's API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_type: str,
        api_version: str,
        api_base: str,
        stop_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the OpenAI API.

        Args:
            model_name (str): The name of the model to use.
            api_key (str): The API key to use.
            api_type (str): The API type to use.
            api_version (str): The API version to use.
            api_base (str): The API base to use.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            None
        """
        # Set the OpenAI API parameters
        self.model_name = model_name
        self.api_key = api_key
        self.api_type = api_type
        self.api_version = api_version
        self.api_base = api_base
        self.stop_tokens = stop_tokens

        # Set the model type
        self.model_type = "chat"

        # Set the client to None (by default)
        self.client = None

        ## OLD AZURE API CODE
        if self.api_type == "azure":
            openai.api_key = self.api_key
            openai.api_type = self.api_type
            openai.api_version = self.api_version
            openai.api_base = self.api_base

            if self.model_name in [
                "code-davinci-002",
                "text-davinci-002",
                "text-davinci-003",
            ]:
                self.model_type = "completion"
            elif self.model_name in [
                "gpt-4",
                "gpt-4-32k",
                "gpt-35-turbo",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-35-turbo-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                "gpt-35-turbo-0613",
                "gpt-35-turbo",
            ]:
                self.model_type = "chat"
            else:
                raise ValueError(f"Model {self.model_name} not supported.")
        else:
            ## NEW OPENAI API CODE SUPPORT
            # Skipping model name validation for now
            # Set up the client
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

    @retry.retry(tries=3, delay=1)
    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate text based on a prompt or messages.

        Args:
            prompt_or_messages (Union[str, List[Dict[str, str]]]): The prompt or messages to generate text from.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.
            temperature (float, optional): The temperature to use. Defaults to 0.7.
            top_p (float, optional): The top p to use. Defaults to 1.0.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[str]: The list of generated texts based on the prompt or messages.
        """
        # Set the stop tokens
        stop_tokens = stop_tokens or self.stop_tokens

        print(f"Calling the language model with prompt: {prompt_or_messages}")

        if self.api_type == "azure":
            ## OLD AZURE API CODE
            # Generate the text
            if self.model_type == "chat":
                response = openai.ChatCompletion.create(
                    engine=self.model_name,
                    messages=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of messages
                return [message["message"]["content"] for message in response.choices]
            else:
                response = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of outputs
                return [output["text"] for output in response.choices]
        else:
            ## NEW OPENAI API CODE SUPPORT
            # Generate the text
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_or_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_return_sequences,
                stop=stop_tokens,
                **kwargs,
            )
            return [output.message.content for output in response.choices]



from typing import List, Union, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class xlam_LanguageModel:
    """A language model using Salesforce's xLAM models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the xLAM model.

        Args:
            model_name (str): The name of the xLAM model to use.
            device (Optional[str], optional): The device to run the model on ('cpu' or 'cuda'). Defaults to None.
            stop_tokens (Optional[List[str]], optional): Tokens to stop generation. Defaults to None.
        """
        self.model_name = model_name
        self.stop_tokens = stop_tokens

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate text based on a prompt.

        Args:
            prompt (str): The input text prompt.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
            num_return_sequences (int, optional): Number of sequences to return. Defaults to 1.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            top_p (float, optional): Nucleus sampling probability. Defaults to 1.0.
            **kwargs (Any): Additional generation parameters.

        Returns:
            List[str]: Generated text sequences.
        """
        # Encode the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text
        output_sequences = self.model.generate(
            input_ids=inputs['input_ids'],
            max_length=inputs['input_ids'].shape[1] + max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        # Decode the generated sequences
        generated_texts = [
            self.tokenizer.decode(output_sequence, skip_special_tokens=True)
            for output_sequence in output_sequences
        ]

        # Apply stop tokens if provided
        if self.stop_tokens:
            generated_texts = [
                self._apply_stop_tokens(text) for text in generated_texts
            ]

        return generated_texts

    def _apply_stop_tokens(self, text: str) -> str:
        """
        Truncate the text at the first occurrence of any stop token.

        Args:
            text (str): The generated text.

        Returns:
            str: Truncated text.
        """
        for stop_token in self.stop_tokens:
            stop_index = text.find(stop_token)
            if stop_index != -1:
                return text[:stop_index]
        return text