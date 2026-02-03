"""GCS downloader for SanctSound audio files."""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from google.cloud import storage
from tqdm import tqdm

from whalesound_cluster.utils.logging import setup_logging

logger = setup_logging()


class GCSDownloader:
    """Download audio files from Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        output_dir: str = "data/downloads",
        method: str = "python",
    ):
        """
        Initialize GCS downloader.
        
        Args:
            bucket_name: GCS bucket name
            prefix: Prefix path within bucket
            output_dir: Local output directory
            method: "python" or "gsutil"
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method = method
        
        if method == "python":
            try:
                self.client = storage.Client.create_anonymous_client()
            except Exception as e:
                logger.warning(
                    f"Failed to create anonymous client: {e}. "
                    "Falling back to Application Default Credentials."
                )
                self.client = storage.Client()
        else:
            self.client = None

    def list_files(
        self,
        max_files: Optional[int] = None,
        date_range: Optional[dict] = None,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """
        List files in GCS bucket matching criteria.
        
        Args:
            max_files: Maximum number of files to return
            date_range: Dict with "start" and "end" dates (YYYY-MM-DD)
            pattern: Regex pattern to match filenames
            
        Returns:
            List of GCS object paths
        """
        if self.method == "gsutil":
            return self._list_files_gsutil(max_files, date_range, pattern)
        
        # Python method
        bucket = self.client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.prefix)
        
        files = []
        for blob in blobs:
            # Accept both .wav and .flac files
            if not (blob.name.endswith(".wav") or blob.name.endswith(".flac")):
                continue
            
            # Check pattern
            if pattern and not re.search(pattern, blob.name):
                continue
            
            # Check date range (extract from filename if possible)
            if date_range:
                file_date = self._extract_date_from_path(blob.name)
                if file_date:
                    start = datetime.strptime(date_range["start"], "%Y-%m-%d")
                    end = datetime.strptime(date_range["end"], "%Y-%m-%d")
                    if not (start <= file_date <= end):
                        continue
            
            files.append(blob.name)
            
            if max_files and len(files) >= max_files:
                break
        
        logger.info(f"Found {len(files)} files matching criteria")
        return files

    def _list_files_gsutil(
        self,
        max_files: Optional[int] = None,
        date_range: Optional[dict] = None,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List files using gsutil command."""
        gs_path = f"gs://{self.bucket_name}/{self.prefix}"
        
        try:
            cmd = ["gsutil", "ls", gs_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            files = [line.strip() for line in result.stdout.split("\n") if line.strip()]
            files = [f for f in files if f.endswith(".wav") or f.endswith(".flac")]
            
            # Apply filters
            if pattern:
                files = [f for f in files if re.search(pattern, f)]
            
            if date_range:
                filtered = []
                for f in files:
                    file_date = self._extract_date_from_path(f)
                    if file_date:
                        start = datetime.strptime(date_range["start"], "%Y-%m-%d")
                        end = datetime.strptime(date_range["end"], "%Y-%m-%d")
                        if start <= file_date <= end:
                            filtered.append(f)
                files = filtered
            
            if max_files:
                files = files[:max_files]
            
            logger.info(f"Found {len(files)} files using gsutil")
            return files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"gsutil command failed: {e}")
            raise
        except FileNotFoundError:
            logger.error("gsutil not found. Install it or use method='python'")
            raise

    def _extract_date_from_path(self, path: str) -> Optional[datetime]:
        """Extract date from filename/path if possible."""
        # Common patterns: YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD
        patterns = [
            r"(\d{4})(\d{2})(\d{2})",
            r"(\d{4})-(\d{2})-(\d{2})",
            r"(\d{4})_(\d{2})_(\d{2})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                try:
                    year, month, day = match.groups()
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None

    def download_file(
        self, 
        gcs_path: str, 
        local_path: Optional[Path] = None,
        skip_existing: bool = False,
    ) -> Optional[Path]:
        """
        Download a single file from GCS.
        
        Args:
            gcs_path: Full GCS path (gs://bucket/path) or blob path within bucket
            local_path: Local destination path
            skip_existing: If True, skip download if file already exists
            
        Returns:
            Path to downloaded file, or None if skipped
        """
        # Determine bucket name and blob path
        if gcs_path.startswith("gs://"):
            # Full GCS path: gs://bucket/path
            gcs_path = gcs_path[5:]  # Remove gs://
            if "/" not in gcs_path:
                raise ValueError(f"Invalid GCS path: gs://{gcs_path}")
            bucket_name, blob_path = gcs_path.split("/", 1)
        elif "/" in gcs_path and gcs_path.split("/")[0] == self.bucket_name:
            # Path starts with bucket name: bucket/path
            bucket_name, blob_path = gcs_path.split("/", 1)
        else:
            # Just blob path (from list_files): use self.bucket_name
            bucket_name = self.bucket_name
            blob_path = gcs_path
        
        if local_path is None:
            # Create local path preserving structure
            rel_path = blob_path.replace(self.prefix, "").lstrip("/")
            local_path = self.output_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"File already exists, skipping: {local_path}")
            return local_path
        
        if self.method == "gsutil":
            # Reconstruct full GCS path for gsutil
            full_gcs_path = f"gs://{bucket_name}/{blob_path}"
            return self._download_file_gsutil(full_gcs_path, local_path)
        
        # Python method
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Check if blob exists before attempting download
        if not blob.exists():
            raise FileNotFoundError(
                f"Blob does not exist: gs://{bucket_name}/{blob_path}"
            )
        
        try:
            blob.download_to_filename(str(local_path))
            # Verify file was actually downloaded (not empty)
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()  # Remove empty file
                raise ValueError(f"Downloaded file is empty: {local_path}")
        except Exception as e:
            # Clean up empty file if it was created
            if local_path.exists() and local_path.stat().st_size == 0:
                local_path.unlink()
            raise
        
        return local_path

    def _download_file_gsutil(self, gcs_path: str, local_path: Path) -> Path:
        """Download file using gsutil."""
        if not gcs_path.startswith("gs://"):
            gcs_path = f"gs://{gcs_path}"
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ["gsutil", "cp", gcs_path, str(local_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return local_path

    def download_batch(
        self,
        files: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        date_range: Optional[dict] = None,
        pattern: Optional[str] = None,
        skip_existing: bool = False,
    ) -> List[Path]:
        """
        Download multiple files.
        
        Args:
            files: Pre-computed list of files. If None, will list files.
            max_files: Maximum files to download (new files only if skip_existing=True)
            date_range: Date range filter
            pattern: Filename pattern filter
            skip_existing: If True, skip download if file already exists
            
        Returns:
            List of downloaded file paths (including existing files if skip_existing=True)
        """
        if files is None:
            # If skip_existing is True and max_files is set, we need to filter out
            # existing files first, then apply max_files to get the next N new files
            if skip_existing and max_files is not None:
                # List files without max_files limit first
                all_files = self.list_files(max_files=None, date_range=date_range, pattern=pattern)
                
                # Filter out files that already exist locally
                new_files = []
                skipped_files = []
                for gcs_path in all_files:
                    # Determine local path (list_files returns blob paths)
                    # Use the same logic as download_file for consistency
                    rel_path = gcs_path.replace(self.prefix, "").lstrip("/")
                    local_path = self.output_dir / rel_path
                    
                    # Check if file exists and is not empty
                    if local_path.exists() and local_path.stat().st_size > 0:
                        skipped_files.append(local_path)
                        # Log first few skipped files at INFO level, rest at DEBUG
                        if len(skipped_files) <= 5:
                            logger.info(f"Skipping existing file: {local_path.name}")
                        else:
                            logger.debug(f"Skipping existing file: {local_path.name}")
                    else:
                        new_files.append(gcs_path)
                    
                    # Stop once we have enough new files
                    if len(new_files) >= max_files:
                        break
                
                files = new_files
                skipped_count = len(skipped_files)
                if skipped_count > 0:
                    logger.info(
                        f"Filtered to {len(files)} new files (skipping {skipped_count} existing files)"
                    )
                    if skipped_count > 5:
                        logger.debug(
                            f"Skipped {skipped_count - 5} additional existing files "
                            f"(use DEBUG logging level to see all skipped files)"
                        )
            else:
                # Normal behavior: list files with max_files limit
                files = self.list_files(max_files=max_files, date_range=date_range, pattern=pattern)
        
        downloaded = []
        skipped = []
        failed = []
        
        # Use tqdm with postfix to show success/failure counts
        pbar = tqdm(files, desc="Downloading files")
        for gcs_path in pbar:
            try:
                # Determine local path for existence check
                # Use the same logic as download_file
                if "/" in gcs_path and not gcs_path.startswith("gs://"):
                    rel_path = gcs_path.replace(self.prefix, "").lstrip("/")
                    local_path = self.output_dir / rel_path
                else:
                    # download_file will determine the path
                    local_path = None
                
                # Check if file exists before attempting download
                if skip_existing:
                    if local_path is not None and local_path.exists() and local_path.stat().st_size > 0:
                        downloaded.append(local_path)
                        skipped.append(local_path)
                        logger.debug(f"Skipping existing file: {local_path.name}")
                        pbar.set_postfix({
                            "new": len(downloaded) - len(skipped),
                            "skipped": len(skipped),
                            "failed": len(failed)
                        })
                        continue
                
                local_path = self.download_file(gcs_path, skip_existing=skip_existing)
                if local_path is not None and local_path not in downloaded:
                    downloaded.append(local_path)
                pbar.set_postfix({
                    "new": len(downloaded) - len(skipped),
                    "skipped": len(skipped),
                    "failed": len(failed)
                })
            except Exception as e:
                failed.append(gcs_path)
                logger.warning(f"Failed to download {gcs_path}: {e}")
                pbar.set_postfix({
                    "new": len(downloaded) - len(skipped),
                    "skipped": len(skipped),
                    "failed": len(failed)
                })
                continue
        
        pbar.close()
        
        new_downloads = len(downloaded) - len(skipped)
        logger.info(
            f"Download complete: {new_downloads} newly downloaded, "
            f"{len(skipped)} already existed, {len(failed)} failed "
            f"out of {len(files)} total files"
        )
        if skipped:
            # Show sample of skipped files
            sample_size = min(5, len(skipped))
            skipped_sample = [p.name for p in skipped[:sample_size]]
            logger.info(f"Skipped files (showing {sample_size} of {len(skipped)}): {', '.join(skipped_sample)}")
            if len(skipped) > sample_size:
                logger.debug(f"Additional {len(skipped) - sample_size} skipped files (use DEBUG logging to see all)")
        if downloaded:
            logger.info(f"Total files available: {len(downloaded)} in {self.output_dir}")
        if failed:
            logger.warning(
                f"{len(failed)} files failed to download. "
                "Check GCS bucket access and file paths."
            )
            # Show sample of failed files
            sample_size = min(5, len(failed))
            failed_sample = failed[:sample_size]
            logger.warning(f"Failed files (showing {sample_size} of {len(failed)}): {', '.join(failed_sample)}")
        
        return downloaded

