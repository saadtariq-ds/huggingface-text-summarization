from src.huggingface_text_summarization.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.huggingface_text_summarization.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.huggingface_text_summarization.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.huggingface_text_summarization.logging import logger


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20}")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20}")
except Exception as e:
    logger.exception(f"Error occurred in {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20}")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20}")
except Exception as e:
    logger.exception(f"Error occurred in {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Model Training Stage"

try:
    logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20}")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.initiate_model_training()
    logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20}")
except Exception as e:
    logger.exception(f"Error occurred in {STAGE_NAME}: {e}")
    raise e


