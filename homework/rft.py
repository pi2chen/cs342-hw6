import json
from pathlib import Path

from .base_llm import BaseLLM
from .sft import test_model, tokenize, TokenizedDataset


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> RFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


# The following code (rft.py:33-45) was written by Claude Opus 4.5.
class RFTDataset:
    """Dataset for RFT that loads from data/rft.json"""
    def __init__(self):
        data_path = Path(__file__).parent.parent / "data" / "rft.json"
        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return (question, answer, reasoning)
        return self.data[idx]


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format an RFT data element.
    The reasoning already contains the chain-of-thought and <answer>...</answer> tags.
    """
    return {"question": question, "answer": reasoning}


def train_model(
    output_dir: str = "./homework/rft_model",
    **kwargs,
):
    # The following code (rft.py:61-108) was written by Claude Opus 4.5.
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = RFTModel()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)

    llm.model.enable_input_require_grads()

    llm.model.print_trainable_parameters()

    train_data = RFTDataset()
    tokenized_train = TokenizedDataset(llm.tokenizer, train_data, format_rft_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=7e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_train,
    )

    trainer.train()

    llm.model.save_pretrained(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
