# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distributed Storage Backend for Recordings.

This module provides storage backends for browser recordings in different deployment modes:
- LocalStorage: File-based storage for embedded/standalone modes
- S3Storage: AWS S3/MinIO storage for cluster deployments
- SharedStorage: NFS/shared filesystem for cluster deployments

Recordings metadata is tracked in the Raft state machine for cluster consensus.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from flybrowser.utils.logger import logger


class StorageBackend(str, Enum):
    """Supported storage backends."""
    
    LOCAL = "local"         # Local filesystem
    S3 = "s3"               # AWS S3 / MinIO
    SHARED = "shared"       # Shared filesystem (NFS, etc.)


@dataclass
class RecordingInfo:
    """Information about a stored recording.
    
    Attributes:
        recording_id: Unique recording identifier
        session_id: Browser session ID
        node_id: Node that created the recording
        file_name: Original file name
        file_size_bytes: File size in bytes
        duration_seconds: Recording duration
        codec: Video codec used
        width: Video width
        height: Video height
        frame_rate: Frames per second
        created_at: Creation timestamp
        storage_backend: Storage backend used
        storage_path: Path/key in storage backend
        download_url: Optional presigned download URL
        metadata: Additional metadata
    """
    
    recording_id: str
    session_id: str
    node_id: str
    file_name: str
    file_size_bytes: int
    duration_seconds: float
    codec: str
    width: int
    height: int
    frame_rate: int
    created_at: float
    storage_backend: StorageBackend
    storage_path: str
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["storage_backend"] = self.storage_backend.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RecordingInfo:
        """Create from dictionary."""
        if "storage_backend" in data and isinstance(data["storage_backend"], str):
            data["storage_backend"] = StorageBackend(data["storage_backend"])
        return cls(**data)


class RecordingStorage(ABC):
    """Abstract base class for recording storage backends."""
    
    @abstractmethod
    async def store(
        self,
        recording_id: str,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> RecordingInfo:
        """Store a recording file.
        
        Args:
            recording_id: Unique recording identifier
            file_path: Local path to recording file
            metadata: Recording metadata
            
        Returns:
            RecordingInfo with storage details
        """
        pass
    
    @abstractmethod
    async def retrieve(self, recording_id: str) -> Optional[RecordingInfo]:
        """Retrieve recording metadata.
        
        Args:
            recording_id: Recording identifier
            
        Returns:
            RecordingInfo if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def download(self, recording_id: str, output_path: str) -> bool:
        """Download a recording file.
        
        Args:
            recording_id: Recording identifier
            output_path: Local path to save file
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, recording_id: str) -> bool:
        """Delete a recording.
        
        Args:
            recording_id: Recording identifier
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def list(
        self,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RecordingInfo]:
        """List recordings with optional filters.
        
        Args:
            session_id: Filter by session ID
            node_id: Filter by node ID
            limit: Maximum number of results
            
        Returns:
            List of RecordingInfo objects
        """
        pass
    
    @abstractmethod
    async def cleanup_old(self, older_than_days: int) -> int:
        """Clean up recordings older than specified days.
        
        Args:
            older_than_days: Delete recordings older than this many days
            
        Returns:
            Number of recordings deleted
        """
        pass


class LocalStorage(RecordingStorage):
    """Local filesystem storage backend.
    
    Stores recordings in a local directory with metadata in JSON files.
    Suitable for embedded and standalone modes.
    """
    
    def __init__(self, base_dir: str = "./recordings") -> None:
        """Initialize local storage.
        
        Args:
            base_dir: Base directory for storing recordings
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.base_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        logger.info(f"LocalStorage initialized at {self.base_dir}")
    
    async def store(
        self,
        recording_id: str,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> RecordingInfo:
        """Store recording locally."""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Recording file not found: {file_path}")
        
        # Copy file to storage directory
        dest_path = self.base_dir / f"{recording_id}{source_path.suffix}"
        await asyncio.to_thread(shutil.copy2, source_path, dest_path)
        
        # Create recording info
        file_size = dest_path.stat().st_size
        info = RecordingInfo(
            recording_id=recording_id,
            session_id=metadata.get("session_id", ""),
            node_id=metadata.get("node_id", "local"),
            file_name=dest_path.name,
            file_size_bytes=file_size,
            duration_seconds=metadata.get("duration_seconds", 0.0),
            codec=metadata.get("codec", "unknown"),
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            frame_rate=metadata.get("frame_rate", 0),
            created_at=time.time(),
            storage_backend=StorageBackend.LOCAL,
            storage_path=str(dest_path),
            metadata=metadata,
        )
        
        # Save metadata
        await self._save_metadata(info)
        
        logger.info(f"Stored recording {recording_id} locally ({file_size} bytes)")
        return info
    
    async def retrieve(self, recording_id: str) -> Optional[RecordingInfo]:
        """Retrieve recording metadata."""
        metadata_path = self.metadata_dir / f"{recording_id}.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            data = await asyncio.to_thread(metadata_path.read_text)
            return RecordingInfo.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to retrieve recording metadata: {e}")
            return None
    
    async def download(self, recording_id: str, output_path: str) -> bool:
        """Download (copy) recording file."""
        info = await self.retrieve(recording_id)
        if not info:
            return False
        
        source_path = Path(info.storage_path)
        if not source_path.exists():
            logger.error(f"Recording file not found: {source_path}")
            return False
        
        try:
            await asyncio.to_thread(shutil.copy2, source_path, output_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download recording: {e}")
            return False
    
    async def delete(self, recording_id: str) -> bool:
        """Delete recording and metadata."""
        info = await self.retrieve(recording_id)
        if not info:
            return False
        
        try:
            # Delete file
            file_path = Path(info.storage_path)
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
            
            # Delete metadata
            metadata_path = self.metadata_dir / f"{recording_id}.json"
            if metadata_path.exists():
                await asyncio.to_thread(metadata_path.unlink)
            
            logger.info(f"Deleted recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete recording: {e}")
            return False
    
    async def list(
        self,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RecordingInfo]:
        """List recordings with filters."""
        recordings = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            if len(recordings) >= limit:
                break
            
            try:
                data = await asyncio.to_thread(metadata_file.read_text)
                info = RecordingInfo.from_dict(json.loads(data))
                
                # Apply filters
                if session_id and info.session_id != session_id:
                    continue
                if node_id and info.node_id != node_id:
                    continue
                
                recordings.append(info)
                
            except Exception as e:
                logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
        
        # Sort by creation time (newest first)
        recordings.sort(key=lambda x: x.created_at, reverse=True)
        return recordings
    
    async def cleanup_old(self, older_than_days: int) -> int:
        """Clean up old recordings."""
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        deleted_count = 0
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                data = await asyncio.to_thread(metadata_file.read_text)
                info = RecordingInfo.from_dict(json.loads(data))
                
                if info.created_at < cutoff_time:
                    if await self.delete(info.recording_id):
                        deleted_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process {metadata_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old recordings")
        return deleted_count
    
    async def _save_metadata(self, info: RecordingInfo) -> None:
        """Save recording metadata to JSON file."""
        metadata_path = self.metadata_dir / f"{info.recording_id}.json"
        data = json.dumps(info.to_dict(), indent=2)
        await asyncio.to_thread(metadata_path.write_text, data)


class S3Storage(RecordingStorage):
    """S3-compatible storage backend (AWS S3, MinIO, etc.).
    
    Stores recordings in S3 with metadata as object tags or separate JSON objects.
    Suitable for cluster deployments requiring shared storage.
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        prefix: str = "recordings/",
    ) -> None:
        """Initialize S3 storage.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            endpoint_url: Custom endpoint URL (for MinIO, etc.)
            access_key: AWS access key (uses env vars if not provided)
            secret_key: AWS secret key (uses env vars if not provided)
            prefix: Key prefix for recordings
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.prefix = prefix.rstrip("/") + "/"
        
        # Import boto3 only when needed
        try:
            import boto3
            from botocore.config import Config
            
            # Create S3 client
            config = Config(region_name=region, signature_version="s3v4")
            
            kwargs = {"config": config}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if access_key and secret_key:
                kwargs["aws_access_key_id"] = access_key
                kwargs["aws_secret_access_key"] = secret_key
            
            self.s3_client = boto3.client("s3", **kwargs)
            
            # Verify bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket)
                logger.info(f"S3Storage initialized (bucket: {bucket}, region: {region})")
            except Exception as e:
                logger.warning(f"S3 bucket verification failed: {e}")
                
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )
    
    async def store(
        self,
        recording_id: str,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> RecordingInfo:
        """Store recording in S3."""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Recording file not found: {file_path}")
        
        # S3 key for recording
        key = f"{self.prefix}{recording_id}{source_path.suffix}"
        
        # Upload file
        file_size = source_path.stat().st_size
        
        def _upload():
            self.s3_client.upload_file(
                str(source_path),
                self.bucket,
                key,
                ExtraArgs={"Metadata": {"recording_id": recording_id}},
            )
        
        await asyncio.to_thread(_upload)
        
        # Create recording info
        info = RecordingInfo(
            recording_id=recording_id,
            session_id=metadata.get("session_id", ""),
            node_id=metadata.get("node_id", "unknown"),
            file_name=source_path.name,
            file_size_bytes=file_size,
            duration_seconds=metadata.get("duration_seconds", 0.0),
            codec=metadata.get("codec", "unknown"),
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            frame_rate=metadata.get("frame_rate", 0),
            created_at=time.time(),
            storage_backend=StorageBackend.S3,
            storage_path=key,
            metadata=metadata,
        )
        
        # Store metadata as separate JSON object
        metadata_key = f"{self.prefix}.metadata/{recording_id}.json"
        await self._upload_json(metadata_key, info.to_dict())
        
        logger.info(f"Stored recording {recording_id} in S3 ({file_size} bytes)")
        return info
    
    async def retrieve(self, recording_id: str) -> Optional[RecordingInfo]:
        """Retrieve recording metadata from S3."""
        metadata_key = f"{self.prefix}.metadata/{recording_id}.json"
        
        try:
            data = await self._download_json(metadata_key)
            return RecordingInfo.from_dict(data)
        except Exception as e:
            logger.debug(f"Failed to retrieve recording metadata: {e}")
            return None
    
    async def download(self, recording_id: str, output_path: str) -> bool:
        """Download recording from S3."""
        info = await self.retrieve(recording_id)
        if not info:
            return False
        
        try:
            def _download():
                self.s3_client.download_file(self.bucket, info.storage_path, output_path)
            
            await asyncio.to_thread(_download)
            logger.info(f"Downloaded recording {recording_id} from S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download recording from S3: {e}")
            return False
    
    async def delete(self, recording_id: str) -> bool:
        """Delete recording from S3."""
        info = await self.retrieve(recording_id)
        if not info:
            return False
        
        try:
            # Delete recording file
            def _delete():
                self.s3_client.delete_object(Bucket=self.bucket, Key=info.storage_path)
            
            await asyncio.to_thread(_delete)
            
            # Delete metadata
            metadata_key = f"{self.prefix}.metadata/{recording_id}.json"
            await asyncio.to_thread(
                self.s3_client.delete_object, Bucket=self.bucket, Key=metadata_key
            )
            
            logger.info(f"Deleted recording {recording_id} from S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete recording from S3: {e}")
            return False
    
    async def list(
        self,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[RecordingInfo]:
        """List recordings from S3."""
        recordings = []
        metadata_prefix = f"{self.prefix}.metadata/"
        
        try:
            def _list():
                paginator = self.s3_client.get_paginator("list_objects_v2")
                return paginator.paginate(
                    Bucket=self.bucket,
                    Prefix=metadata_prefix,
                    PaginationConfig={"MaxItems": limit},
                )
            
            pages = await asyncio.to_thread(_list)
            
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    
                    try:
                        data = await self._download_json(key)
                        info = RecordingInfo.from_dict(data)
                        
                        # Apply filters
                        if session_id and info.session_id != session_id:
                            continue
                        if node_id and info.node_id != node_id:
                            continue
                        
                        recordings.append(info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read metadata from {key}: {e}")
            
            # Sort by creation time (newest first)
            recordings.sort(key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list recordings from S3: {e}")
        
        return recordings
    
    async def cleanup_old(self, older_than_days: int) -> int:
        """Clean up old recordings from S3."""
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        deleted_count = 0
        
        recordings = await self.list(limit=1000)
        
        for info in recordings:
            if info.created_at < cutoff_time:
                if await self.delete(info.recording_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old recordings from S3")
        return deleted_count
    
    async def get_presigned_url(
        self,
        recording_id: str,
        expiration: int = 3600,
    ) -> Optional[str]:
        """Generate presigned URL for downloading recording.
        
        Args:
            recording_id: Recording identifier
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL if successful
        """
        info = await self.retrieve(recording_id)
        if not info:
            return None
        
        try:
            def _generate_url():
                return self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": info.storage_path},
                    ExpiresIn=expiration,
                )
            
            url = await asyncio.to_thread(_generate_url)
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    async def _upload_json(self, key: str, data: Dict[str, Any]) -> None:
        """Upload JSON data to S3."""
        json_bytes = json.dumps(data).encode("utf-8")
        
        def _upload():
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_bytes,
                ContentType="application/json",
            )
        
        await asyncio.to_thread(_upload)
    
    async def _download_json(self, key: str) -> Dict[str, Any]:
        """Download and parse JSON from S3."""
        def _download():
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        
        data = await asyncio.to_thread(_download)
        return json.loads(data)


class SharedStorage(LocalStorage):
    """Shared filesystem storage backend (NFS, etc.).
    
    Similar to LocalStorage but assumes a shared filesystem accessible
    from all cluster nodes. Uses file locking for concurrent access.
    """
    
    def __init__(self, base_dir: str) -> None:
        """Initialize shared storage.
        
        Args:
            base_dir: Base directory on shared filesystem
        """
        super().__init__(base_dir)
        logger.info(f"SharedStorage initialized at {self.base_dir}")
    
    # Inherits all methods from LocalStorage
    # In a production environment, you would add file locking
    # to prevent concurrent access issues


def create_storage_backend(
    backend: StorageBackend,
    **config: Any,
) -> RecordingStorage:
    """Factory function to create storage backend.
    
    Args:
        backend: Storage backend type
        **config: Backend-specific configuration
        
    Returns:
        RecordingStorage instance
    """
    if backend == StorageBackend.LOCAL:
        return LocalStorage(base_dir=config.get("base_dir", "./recordings"))
    
    elif backend == StorageBackend.S3:
        return S3Storage(
            bucket=config["bucket"],
            region=config.get("region", "us-east-1"),
            endpoint_url=config.get("endpoint_url"),
            access_key=config.get("access_key"),
            secret_key=config.get("secret_key"),
            prefix=config.get("prefix", "recordings/"),
        )
    
    elif backend == StorageBackend.SHARED:
        return SharedStorage(base_dir=config["base_dir"])
    
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
