import os
import urllib.request as request
import zipfile
from src.huggingface_text_summarization.entity import DataIngestionConfig
from src.huggingface_text_summarization.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, _ = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Data is Downloaded Successfully")
        else:
            logger.info(f"Data is Already Existed")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_directory
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, mode="r") as zip_file:
            zip_file.extractall(unzip_path)
