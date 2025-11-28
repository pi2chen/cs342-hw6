def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate a dataset by sampling multiple completions from CoTModel and selecting correct ones.
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Number of completions to generate per question (10-20 recommended)
        temperature: Sampling temperature for generation
    """
    import json
    from tqdm import tqdm
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    # Load the CoT model and training dataset
    model = CoTModel()
    train_data = Dataset("train")
    
    generated_data = []
    
    # Process each question in the training set
    for idx in tqdm(range(len(train_data)), desc="Generating dataset"):
        question, correct_answer = train_data[idx]
        
        # Generate multiple completions with temperature sampling
        completions = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]  # [0] because we pass a single question, get list of completions
        
        # Check each completion for correctness
        correct_completion = None
        for completion in completions:
            parsed_answer = model.parse_answer(completion)
            
            # Check if this answer is valid (within tolerance)
            if not (parsed_answer != parsed_answer):  # Check not NaN
                if is_answer_valid(parsed_answer, correct_answer):
                    correct_completion = completion
                    break
        
        # If we found a correct completion, add it to the dataset
        # Format: [question, correct_answer, completion]
        if correct_completion is not None:
            generated_data.append([question, correct_answer, correct_completion])
    
    # Save the generated dataset
    # print(f"Generated {len(generated_data)} examples out of {len(train_data)} questions")
    with open(output_json, "w") as f:
        json.dump(generated_data, f, indent=2)
    
    # print(f"Dataset saved to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
