import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
    TrainingArguments, DataCollatorForSeq2Seq
)
from src.huggingface_text_summarization.entity import ModelTrainerConfig
from src.huggingface_text_summarization.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_ckpt
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Loading the Data
        dataset_samsum_pt = load_from_disk(
            self.config.data_path
        )

        # Initializing Training Arguments
        trainer_args = TrainingArguments(
            output_dir='pegasus-samsum',
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        logger.info("Starting the Training")
        trainer = Trainer(
            model=model, args=trainer_args,
            processing_class=tokenizer, data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        logger.info("Saving the Model and Tokenizer")
        ## Saving the Model and Tokenizer
        model.save_pretrained(
            os.path.join(self.config.root_directory, "summarization_model")
        )

        tokenizer.save_pretrained(
            os.path.join(self.config.root_directory, "tokenizer")
        )
        logger.info("Model and Tokenizer Saved")