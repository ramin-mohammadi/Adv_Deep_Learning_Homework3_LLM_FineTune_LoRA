from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        # messages= [
        #     {"role": "system", "content": "You will perform unit conversions and be concise."},
        #     {"role": "user", "content": "Please convert 6 mi/h into in/s."},
        #     {"role": "assistant", "content": "mi/h refers to miles per hour and in/s refers to inches per second. 1 mi/h = 17.6 in/s. So 6 mi/h = 6 * 17.6 = 105.6 in/s<answer>105.6</answer>"},
        #     {"role": "user", "content": "Could you convert 6 quart to its corresponding value in cc?"},
        #     {"role": "assistant", "content": "quart refers to quarts and cc refers to cubic centimeters. 1 quart = 946.35 cc. So 6 quart = 6 * 946.35 = 5678.117676000002 cc<answer>5678.117676000002</answer>"},
        #     {"role": "user", "content": "How much is 9 decades when converted to month?"},
        #     {"role": "assistant", "content": "decades refers to decades and month refers to months. 1 decade = 120 months. So 9 decades = 9.0 * 120.0 = 1080.0 months<answer>1080.0</answer>"},
        #     {"role": "user", "content": question},
        # ]
        messages= [
            {"role": "system", "content": "You will perform unit conversions and be concise."},
            {"role": "user", "content": "Please convert 6 mi/h into in/s."},
            {"role": "assistant", "content": "1 mi/h = 17.6 in/s. 6 * 17.6 =<answer>105.6</answer>"},
            {"role": "user", "content": "Could you convert 6 quart to its corresponding value in cc?"},
            {"role": "assistant", "content": "1 quart = 946.35 cc. 6 * 946.35 =<answer>5678.117676</answer>"},
            {"role": "user", "content": "How much is 9 decades when converted to month?"},
            {"role": "assistant", "content": "1 decade = 120 months. 9.0 * 120.0 =<answer>1080.0</answer>"},
            {"role": "user", "content": question},
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
