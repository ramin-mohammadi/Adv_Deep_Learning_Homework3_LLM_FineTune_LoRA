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
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
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
        self.tokenizer.padding_side = "left"
        if num_return_sequences is None:
            num_return_sequences = 1
            
        """ 
        - tokenize input prompts 
        - NOTE: we don't only using the input_ids frmo the tokenizer output, we also need to use the attention_mask
        - In ReadMe: Generation is almost the same between unbatched and batched versions with the only difference being that self.model.generate take both input_ids (the tokenized input) and attention_mask as input. attention_mask is produced by the tokenizer indicating which inputs have been padded. Simply take entire dictionary outputted by tokenizer() and then input it into self.model.generate() as **inputs
        - The **inputs syntax is a pythonic way to unpack the dictionary into keyword arguments so it can utilize both the .input_ids and attention_mask from tokenizer() output
        - Note below inputs variable is a batch of formatted input prompts
        """
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, 
                                      max_new_tokens=50,
                                      do_sample=temperature > 0,
                                      temperature=temperature,
                                      num_return_sequences=num_return_sequences,
                                      eos_token_id=self.tokenizer.eos_token_id).to(self.device)
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
