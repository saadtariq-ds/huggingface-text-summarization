from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_directory: Path
    source_URL: Path
    local_data_file: Path
    unzip_directory: Path


@dataclass
class DataTransformationConfig:
    root_directory: Path
    data_path: Path
    tokenizer_name: str