import pandas as pd
from pathlib import Path
from prefect import task, flow
from prefect.logging import get_run_logger


@task
def download_taxi_data(year: int, month: int, taxi_type: str = "yellow") -> str:
    """Download taxi data and save locally"""
    logger = get_run_logger()

    # Create data directory
    data_dir = Path("/app/taxi-data")
    data_dir.mkdir(exist_ok=True)

    # Download data
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month:02d}.parquet'
    filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    filepath = data_dir / filename

    if not filepath.exists():
        logger.info(f"Downloading {url}")
        df = pd.read_parquet(url)
        df.to_parquet(filepath)
        logger.info(f"Saved {len(df)} records to {filepath}")
        print(f"Downloaded {len(df)} records for {year}-{month:02d}")
    else:
        logger.info(f"File {filepath} already exists")
        df = pd.read_parquet(filepath)
        print(f"Found existing file with {len(df)} records")

    return str(filepath)


@flow(name="Data Download Flow")
def download_data_flow(year: int = 2023, month: int = 3, taxi_type: str = "yellow"):
    """Flow to download required taxi data"""
    logger = get_run_logger()
    logger.info(f"Starting data download for {taxi_type} taxi {year}-{month:02d}")

    filepath = download_taxi_data(year, month, taxi_type)

    logger.info(f"Data download completed: {filepath}")
    return filepath


if __name__ == "__main__":
    # Can be run standalone for testing
    download_data_flow()
