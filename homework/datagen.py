def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate a dataset by sampling multiple completions from CoTModel and selecting correct ones.
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Number of completions to generate per question (10-20 recommended)
        temperature: Sampling temperature for generation
    """
    # The following code (datagen.py:11-43) was written by Claude Opus 4.5.
    import json
    from tqdm import tqdm
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    model = CoTModel()
    train_data = Dataset("train")
    
    generated_data = []
    
    for idx in tqdm(range(len(train_data)), desc="Generating dataset"):
        question, correct_answer = train_data[idx]
        
        completions = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]
        
        correct_completion = None
        for completion in completions:
            parsed_answer = model.parse_answer(completion)
            
            if not (parsed_answer != parsed_answer):
                if is_answer_valid(parsed_answer, correct_answer):
                    correct_completion = completion
                    break
        
        if correct_completion is not None:
            generated_data.append([question, correct_answer, correct_completion])
    
    with open(output_json, "w") as f:
        json.dump(generated_data, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
