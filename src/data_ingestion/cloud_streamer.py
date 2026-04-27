import os
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class CloudDataStreamer:
    """
    Intelligently caches large datasets from cloud storage (Google Drive / S3) 
    to the local NVMe SSD, preventing out-of-storage errors when dealing with 5TB+ data.
    """
    def __init__(self, cache_dir='data/cloud_cache', max_cache_gb=100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_gb = max_cache_gb
        
    def _get_dir_size_gb(self) -> float:
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('**/*') if f.is_file())
        return total_size / (1024 ** 3)

    def _evict_oldest(self):
        """Removes oldest files if cache limit is exceeded."""
        while self._get_dir_size_gb() > self.max_cache_gb:
            files = sorted(self.cache_dir.glob('*'), key=lambda x: x.stat().st_mtime)
            if not files: break
            logger.info(f"Cache full. Evicting {files[0].name} to save space.")
            files[0].unlink()

    def download_file(self, file_id: str, output_filename: str) -> str:
        """
        Downloads a file from Google Drive using its file_id.
        """
        target_path = self.cache_dir / output_filename
        
        if target_path.exists():
            logger.info(f"File {output_filename} already in cache.")
            return str(target_path)
            
        self._evict_oldest()
        logger.info(f"Downloading {output_filename} from Cloud to local NVMe SSD...")
        
        # Using standard direct download link format for Drive
        url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(url, params={'id': file_id, 'confirm': 't'}, stream=True)
        
        # A simple bypass for Google Drive's large-file virus warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                response = session.get(url, params={'id': file_id, 'confirm': value}, stream=True)
                break
                
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
                
        logger.info(f"Download complete: {target_path}")
        return str(target_path)

    def cleanup(self):
        """Clears the entire cache."""
        for f in self.cache_dir.glob('*'):
            if f.is_file():
                f.unlink()
        logger.info("Cloud cache cleared.")