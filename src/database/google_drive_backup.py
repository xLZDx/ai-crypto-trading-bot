"""
Google Drive Backup — Phase 7.

Pushes parquet partitions marked `archived-eligible` by `RetentionManager`
to a Google Drive folder. Returns the Drive URL so the retention index can
record where each archive lives.

Authentication options (first one that works wins):
  1. Service-account JSON pointed at by env var `GDRIVE_SA_JSON`
  2. OAuth flow via `pydrive2` reading `client_secrets.json` from the
     project root (interactive once, then cached in `mycreds.txt`)

If neither is configured this module logs a friendly warning and returns
False from every operation — the bot keeps running, the retention index
just doesn't transition partitions to `archived=True`.

Install:
    pip install --no-cache-dir google-api-python-client google-auth pydrive2

Usage from code:
    bk = GoogleDriveBackup(root_folder_name="ai-trading-archive")
    if bk.is_available():
        url = bk.upload_partition(local_path, "BTC_USDT/1s/yyyymm=2018-01.tar.gz")
        if url:
            retention_mgr.mark_archived(key, url)
"""
from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GoogleDriveBackup:
    """Thin wrapper over pydrive2 for parquet-partition archival.

    The class is designed to fail-soft: every operation returns a value or
    None (never raises) so the trading loop is never blocked by Drive
    issues.
    """

    def __init__(self, root_folder_name: str = "ai-trading-archive"):
        self.root_folder_name = root_folder_name
        self._drive = None
        self._root_id: str | None = None
        self._init_attempted = False

    # ── Authentication ────────────────────────────────────────────────────

    def _init(self) -> bool:
        if self._drive is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
        except ImportError:
            logger.warning("pydrive2 not installed — Google Drive backup disabled. "
                           "Run: pip install --no-cache-dir pydrive2")
            return False

        gauth = GoogleAuth()
        sa_path = os.getenv("GDRIVE_SA_JSON")
        if sa_path and Path(sa_path).exists():
            try:
                from oauth2client.service_account import ServiceAccountCredentials
                gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    sa_path, ["https://www.googleapis.com/auth/drive"],
                )
                logger.info("[gdrive] Using service-account credentials from %s", sa_path)
            except Exception as exc:
                logger.warning("[gdrive] SA auth failed: %s", exc)
                return False
        else:
            client_secrets = PROJECT_ROOT / "client_secrets.json"
            creds_cache    = PROJECT_ROOT / "data" / "gdrive_creds.txt"
            if not client_secrets.exists():
                logger.warning("[gdrive] No GDRIVE_SA_JSON env var and "
                               "client_secrets.json missing — backup disabled.")
                return False
            try:
                gauth.LoadCredentialsFile(str(creds_cache))
                if gauth.credentials is None:
                    gauth.LocalWebserverAuth()
                elif gauth.access_token_expired:
                    gauth.Refresh()
                else:
                    gauth.Authorize()
                gauth.SaveCredentialsFile(str(creds_cache))
            except Exception as exc:
                logger.warning("[gdrive] OAuth setup failed: %s", exc)
                return False

        self._drive = GoogleDrive(gauth)
        self._root_id = self._ensure_root_folder()
        return self._root_id is not None

    def _ensure_root_folder(self) -> str | None:
        try:
            q = (f"title='{self.root_folder_name}' "
                 "and mimeType='application/vnd.google-apps.folder' "
                 "and trashed=false")
            existing = self._drive.ListFile({"q": q}).GetList()
            if existing:
                return existing[0]["id"]
            folder = self._drive.CreateFile({
                "title":    self.root_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            })
            folder.Upload()
            return folder["id"]
        except Exception as exc:
            logger.warning("[gdrive] could not ensure root folder: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._init() and self._drive is not None

    # ── Upload helpers ────────────────────────────────────────────────────

    @staticmethod
    def tar_gz_partition(part_dir: Path, out_path: Path) -> Path:
        """Tar+gzip a parquet partition directory for upload."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_path, "w:gz") as tar:
            for f in part_dir.glob("*.parquet"):
                tar.add(f, arcname=f.name)
        return out_path

    def upload_file(self, local_path: Path, remote_name: str | None = None) -> str | None:
        """Upload a single file to the root folder, return the shareable URL."""
        if not self.is_available():
            return None
        try:
            f = self._drive.CreateFile({
                "title":   remote_name or local_path.name,
                "parents": [{"id": self._root_id}],
            })
            f.SetContentFile(str(local_path))
            f.Upload()
            try:
                f.InsertPermission({
                    "type":  "anyone",
                    "value": "anyone",
                    "role":  "reader",
                })
            except Exception:
                pass
            return f.get("alternateLink") or f.get("webContentLink")
        except Exception as exc:
            logger.warning("[gdrive] upload failed for %s: %s", local_path, exc)
            return None

    def upload_partition(self, part_dir: Path, archive_name: str) -> str | None:
        """tar-gz a partition dir, upload it, return the URL."""
        if not part_dir.exists():
            return None
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / "data" / "cache")) as tmp:
            archive = Path(tmp) / f"{archive_name}.tar.gz"
            self.tar_gz_partition(part_dir, archive)
            return self.upload_file(archive, remote_name=archive.name)


__all__ = ["GoogleDriveBackup"]
