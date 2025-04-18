from .data import Dataset, is_answer_valid
from .cot import CoTModel
import json

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()
    
    trainset = Dataset("train")
    validset = Dataset("valid")
    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
    aug_train_dataset = generate_dataset_helper(dataset=trainset, model=model, oversample=oversample, temperature=temperature)
    # aug_valid_dataset = generate_dataset_helper(dataset=validset, model=model, dataoversample=oversample, temperature=temperature)
    
    print(f"Augmented train dataset size: {len(aug_train_dataset)}")
    print(f"First 5 entires of augmented dataset:\n{aug_train_dataset[:10]}")  # Print the first 5 entries of the augmented dataset
  
    # Save the dataset to a JSON file
    with open(output_json, "w") as f:
        json.dump(aug_train_dataset, f, indent=2)
    # with open(output_json, "w") as f:
    #     json.dump(aug_valid_dataset, f, indent=2)

    print(f"Dataset saved to {output_json}")
    
def generate_dataset_helper(dataset: Dataset, model: CoTModel, oversample: int = 10, temperature: float = 0.6) -> list[ list[str, float, str] ]:
    questions = [dataset[i][0] for i in range(len(dataset))]
    # Note: dataset[i][0] is the ith question, dataset[i][1] is the ith answer
    
    prompts = [model.format_prompt(q) for q in questions] # calls CoTModel.format_prompt
    #print("PROMPTS:\n", prompts)
    generations = model.batched_generate(prompts=prompts, temperature=temperature, num_return_sequences=oversample)
    #print("GENERATIONS:\n", generations)
    
    # Bc the model is generating oversample # of responses per question, we parse each response and see if the answer falls within the relative tolerance (in data.py, tolerance is +/- 0.05)
    # Select the response with the correct answer, and add it to a dataset. If none of the answer is correct, ignore that data point
    
    aug_dataset = []  # augmented dataset
    idx=0
    # Parse each response and validate the answer. If correct, add datapoint to augmented dataset [question, parsed float answer, generated response with reasoning], otherwise ignore datapoint (ignore the question)
    for question, responses in zip(questions, generations):
        for response in responses:
            answer = model.parse_answer(response)  # Extract the float answer from the generated response
            if is_answer_valid(answer=answer, correct_answer=dataset[idx][1]):  # Check if the answer is "correct" being that it falls within the relative tolerance
                aug_dataset.append([question, answer, response])
                break  # Stop after finding the first valid response
        idx += 1
    return aug_dataset


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
