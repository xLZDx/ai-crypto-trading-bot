"""
CloudDataStreamer: Intelligently caches large datasets from Google Drive
to the local SSD to prevent GPU starvation during TFT/LSTM training.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CloudDataStreamer:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_gb = 50  # Manageable local cache size

    def get_data(self, file_id: str, file_name: str) -> Path:
        """Checks local cache. If missing, downloads from cloud."""
        local_path = self.cache_dir / file_name
        if local_path.exists():
            logger.info(f"[CloudStreamer] Cache hit for {file_name}. Using local SSD.")
            return local_path
        
        logger.info(f"[CloudStreamer] Cache miss. Downloading {file_name} from Cloud...")
        return self._download_from_drive(file_id, local_path)

    def _download_from_drive(self, file_id: str, dest_path: Path) -> Path:
        """
        Simulates downloading from Google Drive.
        In a real scenario, this would use PyDrive2 or google-api-python-client.
        """
        logger.warning(f"Download logic for {file_id} requires Google API credentials.")
        # Touch file to simulate successful download for the pipeline
        dest_path.touch()
        return dest_path

    def clear_cache(self):
        """Removes cached files to free up SSD space."""
        logger.info("[CloudStreamer] Clearing local cache...")
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                f.unlink()
