from .base_llm import BaseLLM
from .sft import TokenizedDataset
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    #llm = BaseLLM()
    # llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype="bfloat16")
    llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype="bfloat16")
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(prompt: str, float_answer: float, answer: str) -> dict[str, str]:
    """
    Create data of question + reasoning pairs. Answer rounded to make it easier for the LLM.
    """
    #raise NotImplementedError()
    
    # this method is called in the TokenizedDataset class and is the format_fn paramater. Look at format_fn's description
    
    # parse float using answer tags, round to 3 decimal places, then join string again with answer tags
    left=answer.split("<answer>")
    right=left[1].split("</answer>")
    answer = left[0] + "<answer>" + str(round(float(right[0]), 3)) + "</answer>"
    
    # answer is the generated resposne from COT so the reasoning with the float answer surrounded by answer tags
    #print("ANSWER:\n", answer)
    
    return {
        "question": prompt,
        "answer": answer,
    }

def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
    
    # Expect: !python -m homework.rft train --output_dir ./homework/rft_model
    
    # llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype="bfloat16")
    llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype="bfloat16")
    
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64, # lora_alpha should be 4 to 5 times r
        bias="none",
        target_modules="all-linear",
        lora_dropout=0.1,
    )
    
    # Load the LLM model with LoRA adapter
    model = get_peft_model(llm.model, peft_config) # llm.model is the variable that contains the model (look in BaseLLM init)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    # Note: bc BaseLLM puts the model (.model) on the gpu, get_peft_model() will also put the model on the gpu 
    
    from transformers import TrainingArguments, Trainer
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=7e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        report_to="tensorboard",
        output_dir=output_dir,
        logging_dir=output_dir,
    )
    # TrainingArguments puts onto gpu if available by default bc use_cpu parameter is False by default, and if so, then moves onto gpu if available. And Trainer() is placed onto device TrainingArguments is on. Also, Trainer() handles internally moving the train and validation datasets to the gpu if available
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(llm.tokenizer, Dataset("rft"), format_example), # Dataset("rft") corresponds to the rft.json we created in datagen.py
        tokenizer=llm.tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    # Note to load our saved model, you will use PeftModel.from_pretrained() bc the lora adpater is a peft model 
    
    test_model(output_dir)
    
def test_model(ckpt_path: str):
    testset = Dataset("valid")
    #llm = BaseLLM()
    #llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype="bfloat16")
    llm = BaseLLM(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype="bfloat16")

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})