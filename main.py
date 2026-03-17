from src.huggingface_text_summarization.pipeline.data_ingestion_pipeline import DataIngestionPipeline
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

