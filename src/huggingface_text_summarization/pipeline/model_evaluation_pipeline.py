from src.huggingface_text_summarization.config.configuration import ConfigurationManager
from src.huggingface_text_summarization.components.model_evaluation import ModelEvaluation
from src.huggingface_text_summarization.logging import logger


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()

        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        model_evaluation.evaluate()
