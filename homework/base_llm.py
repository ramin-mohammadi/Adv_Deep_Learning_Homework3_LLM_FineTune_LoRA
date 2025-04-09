from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # https://github.com/huggingface/smollm/tree/main/text#usage
        messages = [
            {"role": "user", "content": question}
        ]
        """
        Optionally can add a system message to the chat template. -> Added at beginning of message
        messages = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ] 
        """
        
        """
        https://huggingface.co/docs/transformers/main/en/chat_templating 
        
        apply_chat_template() params:
        
        add_generation_prompt: bool
            - indicates the "start of a new message"
            - more specifically it adds the "assistant" role token to the end of the message
            - note the message is referring to as seen by the variable array "messages" above
            - This ensures the chat model generates a system response instead of continuing a users message.
            - but not all models use the assistant role token to indicate to generate a response so setting it to true wont do anything in that case
            - also apparently during training, setting to true does not help
            - if set to false, make sure to set add_special_tokens=False 
            
        continue_final_message: bool
            - The continue_final_message parameter controls whether the final message in the chat should be continued or not instead of starting a new one. It removes end of sequence tokens so that the model continues generation from the final message.
            - This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy for instruction following when you know how to start its replies.

            chat = [
                {"role": "user", "content": "Can you format the answer in JSON?"},
                {"role": "assistant", "content": '{"name": "'},
            ]
            Note in example above, the content of the assistant's message is not complete -> so setting this param to True allows you to prefill/start a model's response and the model will continue from there rather generating a new response from scratch.
        
        tokenize: bool
            - does what you think it does being tokenizes the message by representing the message as a list of token ids -> #s
            - Important to note feeding input into tokenizer model like tokenizer(input) tokenizes as well which we already do in the generate function (output of format_prompt() will be used as input to batch_generate()) so here we dont want to tokenize using apply_chat_template(). The tokenizer model I'm referring to is one like in __init__()      
            
        add_special_tokens: bool
            - Some tokenizers add special <bos> and <eos> tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with apply_chat_template(tokenize=False), make sure you set add_special_tokens=False as well to avoid duplicating them. This isn't an issue if apply_chat_template(tokenize=True).
        """
        # we're not doing training here so can set add_generation_prompt=True since question input will be 
        # individual (its own message and not part of a conversation)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False, add_generation_prompt=True).to(self.device)

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode
 
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
            Left padding makes sure all sequences are aligned to the right  (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        #raise NotImplementedError()
        """
        Example: https://huggingface.co/docs/transformers/main/en/tasks/language_modeling#inference  -> look at Pytorch example
        """
        
        """
        The core structure of batched generation is very similar to regular generation, with one exception: All sequences that go into the transformer need to be of the same length. This is achieved through padding the shorter sequences in the left (aligning all sequences on the right, where generation starts). The transformers library will take care of padding in the self.tokenizer call, simply pass in a list[str] of prompts and use padding=True and return a PyTorch tensor return_tensors="pt"
        """
        self.tokenizer.padding_side = "left"
        
        if num_return_sequences is None:
            num_return_sequences = 1 # this is so dont get error in generate()
            
        """ 
        - tokenize input prompts 
        - NOTE: we don't only using the input_ids frmo the tokenizer output, we also need to use the attention_mask
        - In ReadMe: Generation is almost the same between unbatched and batched versions with the only difference being that self.model.generate take both input_ids (the tokenized input) and attention_mask as input. attention_mask is produced by the tokenizer indicating which inputs have been padded. Simply take entire dictionary outputted by tokenizer() and then input it into self.model.generate() as **inputs
        - The **inputs syntax is a pythonic way to unpack the dictionary into keyword arguments so it can utilize both the .input_ids and attention_mask from tokenizer() output
        - Note below inputs variable is a batch of formatted input prompts
        
        - return_tensors="pt" tells the tokenizer to return the tokenized output as PyTorch tensors. This is useful when working with PyTorch models, as it allows you to directly pass the tokenized inputs to the model without additional conversion.
        - The output will include tensors for input_ids (the tokenized text) and attention_mask (indicating which tokens are padding).
        - input_ids: A tensor containing the tokenized representation of the input text. Each number corresponds to a token ID in the tokenizer's vocabulary.
        - attention_mask: A tensor indicating which tokens are actual input (1) and which are padding (0). This is important for models to ignore padding during processing.
        
        Example: if you print out the result of tokenizer()
        {'input_ids': tensor([[15496,   11,  703,  389,  345,   30]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
        
        Note tokenizer() vs. tokenizer.encode()
        Tokenizer()
            Purpose: Tokenizes input text and returns a dictionary containing multiple components, such as input_ids, attention_mask, and optionally other fields like token_type_ids.
            Output: A dictionary with tokenized data.
            Use Case: When you need more than just the token IDs, such as attention masks or when working with batched inputs.
        Tokenizer.encode()
            Purpose: Tokenizes input text and directly returns the input_ids (a list of token IDs) without additional information like attention_mask.
            Output: A list of token IDs.
            Use Case: When you only need the token IDs and don't require other components like attention masks
        """
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        """
        do_sample: bool
            Purpose: Determines whether the model uses sampling or greedy decoding to generate text.
            Behavior:
            do_sample=True: Enables sampling, where the next token is chosen probabilistically based on the model's predicted distribution. This introduces randomness and allows for more diverse outputs.
            do_sample=False: Disables sampling and uses greedy decoding, where the token with the highest probability is always selected. This produces deterministic outputs but can lead to repetitive or less creative results.
        temperature: float
            Purpose: Controls the randomness of the sampling process when do_sample=True.
            Behavior:
            A higher temperature (e.g., 1.0 or above) increases randomness by making the probability distribution flatter. This allows the model to explore less likely tokens, resulting in more diverse and creative outputs.
            A lower temperature (e.g., 0.1) makes the distribution sharper, favoring tokens with higher probabilities. This reduces randomness and makes the output more focused and deterministic.
            temperature=0: Effectively disables sampling, making the output equivalent to greedy decoding (even if do_sample=True).
            
            Lecture 4.2:
            More or less creative (random) writing by raising model prob to power
            - Temperature T equivalent to multiplying logits with 1/T
        max_new_tokens: 
            The maximum number of tokens to generate. Set this to a reasonable value
        num_return_sequences: 
            The number of sequences to return. Note that this will generate a flat list of len(prompts) * num_return_sequences entries.
        eos_token_id: 
            The end of sequence token id. This is used to stop generation.
            
        You can choose the generation strategy being how the model chooses the next token to generate.
        https://huggingface.co/docs/transformers/main/en/generation_strategies 
        - Lecture 4.2:
            - Top p Nucleus sampling is used everywhere (BEST sampling method)
            - here, if do_sample=False, uses greedy decoding/sampling
        - Greedy search is the default decoding strategy.
        """
        outputs = self.model.generate(**inputs, 
                                      max_new_tokens=50,
                                      do_sample=temperature > 0,
                                      temperature=temperature,
                                      num_return_sequences=num_return_sequences,
                                      eos_token_id=self.tokenizer.eos_token_id).to(self.device)
        """
        mask out the input tokens from the generated output. Here's what it does step by step:

        Explanation
        outputs:
            This is the tensor returned by self.model.generate(). It contains the token IDs for the generated sequences, including both the input tokens (from the prompt) and the newly generated tokens.
        inputs["input_ids"]:
            This is the tensor of token IDs for the input prompts, created by the tokenizer. It represents the tokenized version of the input text.
        len(inputs["input_ids"][0]):
            This calculates the length of the first input sequence (i.e., the number of tokens in the input prompt). This value is used to determine how many tokens in the outputs tensor correspond to the input prompt.
        outputs[:, len(inputs["input_ids"][0]) :]:
            This slices the outputs tensor along the second dimension (token IDs) to exclude the tokens corresponding to the input prompt. It keeps only the tokens generated by the model after the input prompt.
            Why Is This Necessary?
            When generating text, the model appends the generated tokens to the input tokens. If you don't remove the input tokens, the output will include both the input and the generated text. This line ensures that only the newly generated tokens are kept.

        Example
        Input Prompt:
            prompt = "The cat sat on the"
        Tokenized Input (inputs["input_ids"]):
            [15496, 703, 389, 345, 262]
        Model Output (outputs):
            [[15496, 703, 389, 345, 262, 1234, 5678, 91011]]
            # Includes both input tokens and generated tokens
        After Slicing:
            outputs = outputs[:, len(inputs["input_ids"][0]) :]
            # Keeps only the generated tokens
            [[1234, 5678, 91011]]
        Final Output
            The sliced outputs tensor is then decoded into text using the tokenizer, ensuring that the final output only contains the generated text, not the original input prompt.
        """
        outputs = outputs[:, len(inputs["input_ids"][0]) :]
        decode = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # the output of the decode is a flat list of len(prompts) * num_return_sequences entries
        # len(prompts) represents the number of questions/prompts
        # num_return_sequences represents the number of answer responses generated for each question
        if num_return_sequences is None or num_return_sequences == 1:
            return decode
        return decode.reshape(len(prompts), num_return_sequences) 
        # reshape to rows as question and columns as answers for the ith question/row


    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
