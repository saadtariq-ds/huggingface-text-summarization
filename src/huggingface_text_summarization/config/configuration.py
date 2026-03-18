from src.huggingface_text_summarization.entity import DataIngestionConfig, DataTransformationConfig
from src.huggingface_text_summarization.utils.common import read_yaml, create_directories
from src.huggingface_text_summarization.constants import *



class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, param_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(path_to_yaml=config_file_path)
        self.params = read_yaml(path_to_yaml=param_file_path)

        create_directories(path_to_directories=[self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories(path_to_directories=[config.root_directory])

        data_ingestion_config = DataIngestionConfig(
            root_directory=config.root_directory,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_directory=config.unzip_directory
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories(path_to_directories=[config.root_directory])

        data_transformation_config = DataTransformationConfig(
            root_directory=config.root_directory,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
        )

        return data_transformation_config