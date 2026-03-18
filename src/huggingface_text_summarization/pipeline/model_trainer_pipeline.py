from src.huggingface_text_summarization.config.configuration import ConfigurationManager
from src.huggingface_text_summarization.components.model_trainer import ModelTrainer
from src.huggingface_text_summarization.logging import logger


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()

        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)

        model_trainer.train()
