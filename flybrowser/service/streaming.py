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
Live Streaming Infrastructure for FlyBrowser.

This module provides real-time streaming capabilities for browser sessions:
- HLS (HTTP Live Streaming) with adaptive bitrate
- DASH (Dynamic Adaptive Streaming)
- RTMP relay for streaming platforms
- WebSocket-based live streaming
- Stream health monitoring and analytics
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from playwright.async_api import Page

from flybrowser.core.ffmpeg_recorder import (
    FFmpegConfig,
    FFmpegRecorder,
    QualityProfile,
    StreamingProtocol,
    VideoCodec,
)
from flybrowser.utils.logger import logger


class StreamState(str, Enum):
    """State of a streaming session."""
    
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class StreamHealth(str, Enum):
    """Health status of a stream."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class StreamMetrics:
    """Metrics for a streaming session."""
    
    frames_sent: int = 0
    bytes_sent: int = 0
    dropped_frames: int = 0
    current_bitrate: float = 0.0  # bps
    average_bitrate: float = 0.0  # bps
    current_fps: float = 0.0
    viewer_count: int = 0
    buffer_health: float = 100.0  # percentage
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "dropped_frames": self.dropped_frames,
            "current_bitrate": self.current_bitrate,
            "average_bitrate": self.average_bitrate,
            "current_fps": self.current_fps,
            "viewer_count": self.viewer_count,
            "buffer_health": self.buffer_health,
            "last_update": self.last_update,
        }


@dataclass
class StreamConfig:
    """Configuration for a streaming session."""
    
    protocol: StreamingProtocol = StreamingProtocol.HLS
    quality_profile: QualityProfile = QualityProfile.MEDIUM
    codec: VideoCodec = VideoCodec.H264
    width: int = 1280
    height: int = 720
    frame_rate: int = 25
    enable_hw_accel: bool = True
    
    # HLS/DASH specific
    segment_duration: int = 2  # seconds
    playlist_size: int = 10  # number of segments
    
    # RTMP specific
    rtmp_url: Optional[str] = None
    rtmp_key: Optional[str] = None
    
    # Stream limits
    max_viewers: int = 100
    max_duration_seconds: int = 3600  # 1 hour
    
    # Adaptive bitrate
    enable_abr: bool = False  # Adaptive BitRate
    abr_profiles: List[QualityProfile] = field(
        default_factory=lambda: [
            QualityProfile.LOW_BANDWIDTH,
            QualityProfile.MEDIUM,
            QualityProfile.HIGH,
        ]
    )


@dataclass
class StreamInfo:
    """Information about an active stream."""
    
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    node_id: str = ""
    state: StreamState = StreamState.INITIALIZING
    health: StreamHealth = StreamHealth.HEALTHY
    protocol: StreamingProtocol = StreamingProtocol.HLS
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    
    # Stream URLs
    hls_url: Optional[str] = None
    dash_url: Optional[str] = None
    rtmp_url: Optional[str] = None
    websocket_url: Optional[str] = None
    
    # Configuration
    config: StreamConfig = field(default_factory=StreamConfig)
    
    # Metrics
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    
    # Viewers
    viewer_ids: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "state": self.state.value,
            "health": self.health.value,
            "protocol": self.protocol.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "hls_url": self.hls_url,
            "dash_url": self.dash_url,
            "rtmp_url": self.rtmp_url,
            "websocket_url": self.websocket_url,
            "metrics": self.metrics.to_dict(),
            "viewer_count": len(self.viewer_ids),
            "uptime_seconds": time.time() - self.started_at if self.state == StreamState.ACTIVE else 0,
        }


class StreamingSession:
    """Individual streaming session managing a single stream.
    
    Handles the lifecycle of a stream including:
    - FFmpeg process management
    - HLS/DASH playlist generation
    - RTMP relay
    - Metrics collection
    - Viewer management
    """
    
    def __init__(
        self,
        session_id: str,
        config: StreamConfig,
        output_dir: str,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize streaming session.
        
        Args:
            session_id: Browser session ID
            config: Stream configuration
            output_dir: Directory for stream output
            base_url: Base URL for accessing streams
        """
        self.session_id = session_id
        self.config = config
        self.output_dir = Path(output_dir)
        self.base_url = base_url or "http://localhost:8000"
        
        # Create stream directory
        self.stream_dir = self.output_dir / session_id
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        
        # Stream info
        self.info = StreamInfo(
            session_id=session_id,
            protocol=config.protocol,
            config=config,
        )
        
        # FFmpeg recorder
        self._recorder: Optional[FFmpegRecorder] = None
        self._page: Optional[Page] = None
        
        # Monitoring
        self._metrics_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        
        # Viewer tracking
        self._viewer_connections: Dict[str, asyncio.Queue] = {}
        
        logger.info(f"StreamingSession created: {self.info.stream_id}")
    
    async def start(self, page: Page) -> StreamInfo:
        """Start the stream.
        
        Args:
            page: Playwright page to stream
            
        Returns:
            StreamInfo with stream details
        """
        self._page = page
        self.info.state = StreamState.INITIALIZING
        
        try:
            # Build FFmpeg configuration
            ffmpeg_config = self._build_ffmpeg_config()
            
            # Create recorder
            self._recorder = FFmpegRecorder(ffmpeg_config)
            
            # Start recording/streaming
            await self._recorder.start(page)
            
            # Update stream info
            self.info.state = StreamState.ACTIVE
            self._set_stream_urls()
            
            # Start monitoring tasks
            self._metrics_task = asyncio.create_task(self._monitor_metrics())
            self._health_task = asyncio.create_task(self._monitor_health())
            
            logger.info(f"Stream started: {self.info.stream_id}")
            return self.info
            
        except Exception as e:
            self.info.state = StreamState.ERROR
            self.info.health = StreamHealth.UNHEALTHY
            logger.error(f"Failed to start stream: {e}")
            raise
    
    async def stop(self) -> StreamInfo:
        """Stop the stream.
        
        Returns:
            Final StreamInfo with metrics
        """
        if self.info.state == StreamState.STOPPED:
            return self.info
        
        self.info.state = StreamState.STOPPED
        self.info.ended_at = time.time()
        
        # Stop monitoring tasks
        if self._metrics_task:
            self._metrics_task.cancel()
        if self._health_task:
            self._health_task.cancel()
        
        # Stop recorder
        if self._recorder:
            try:
                await self._recorder.stop()
            except Exception as e:
                logger.error(f"Error stopping recorder: {e}")
        
        # Notify viewers
        await self._notify_viewers_stream_ended()
        
        logger.info(f"Stream stopped: {self.info.stream_id}")
        return self.info
    
    async def pause(self) -> bool:
        """Pause the stream."""
        if self.info.state == StreamState.ACTIVE:
            self.info.state = StreamState.PAUSED
            logger.info(f"Stream paused: {self.info.stream_id}")
            return True
        return False
    
    async def resume(self) -> bool:
        """Resume the stream."""
        if self.info.state == StreamState.PAUSED:
            self.info.state = StreamState.ACTIVE
            logger.info(f"Stream resumed: {self.info.stream_id}")
            return True
        return False
    
    async def add_viewer(self, viewer_id: str) -> bool:
        """Add a viewer to the stream.
        
        Args:
            viewer_id: Unique viewer identifier
            
        Returns:
            True if viewer was added
        """
        if len(self.info.viewer_ids) >= self.config.max_viewers:
            logger.warning(f"Stream at max viewers: {self.info.stream_id}")
            return False
        
        self.info.viewer_ids.add(viewer_id)
        self.info.metrics.viewer_count = len(self.info.viewer_ids)
        
        # Create queue for viewer if WebSocket
        if self.config.protocol == StreamingProtocol.HLS:  # WebSocket would need separate handling
            self._viewer_connections[viewer_id] = asyncio.Queue()
        
        logger.info(f"Viewer {viewer_id} joined stream {self.info.stream_id}")
        return True
    
    async def remove_viewer(self, viewer_id: str) -> bool:
        """Remove a viewer from the stream.
        
        Args:
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was removed
        """
        if viewer_id in self.info.viewer_ids:
            self.info.viewer_ids.remove(viewer_id)
            self.info.metrics.viewer_count = len(self.info.viewer_ids)
            
            if viewer_id in self._viewer_connections:
                del self._viewer_connections[viewer_id]
            
            logger.info(f"Viewer {viewer_id} left stream {self.info.stream_id}")
            return True
        return False
    
    def _build_ffmpeg_config(self) -> FFmpegConfig:
        """Build FFmpeg configuration for streaming."""
        output_path = None
        streaming_url = None
        
        if self.config.protocol == StreamingProtocol.HLS:
            output_path = str(self.stream_dir / "playlist.m3u8")
        elif self.config.protocol == StreamingProtocol.DASH:
            output_path = str(self.stream_dir / "manifest.mpd")
        elif self.config.protocol == StreamingProtocol.RTMP:
            if self.config.rtmp_url:
                streaming_url = self.config.rtmp_url
                if self.config.rtmp_key:
                    streaming_url = f"{streaming_url}/{self.config.rtmp_key}"
        
        return FFmpegConfig(
            codec=self.config.codec,
            quality_profile=self.config.quality_profile,
            output_path=output_path,
            streaming_protocol=self.config.protocol,
            streaming_url=streaming_url,
            width=self.config.width,
            height=self.config.height,
            frame_rate=self.config.frame_rate,
            enable_hw_accel=self.config.enable_hw_accel,
        )
    
    def _set_stream_urls(self) -> None:
        """Set stream URLs based on protocol."""
        base = self.base_url.rstrip("/")
        stream_path = f"/streams/{self.info.stream_id}"
        
        if self.config.protocol == StreamingProtocol.HLS:
            self.info.hls_url = f"{base}{stream_path}/playlist.m3u8"
        elif self.config.protocol == StreamingProtocol.DASH:
            self.info.dash_url = f"{base}{stream_path}/manifest.mpd"
        elif self.config.protocol == StreamingProtocol.RTMP:
            self.info.rtmp_url = self.config.rtmp_url
        
        # WebSocket URL (for live updates)
        self.info.websocket_url = f"ws://{base.replace('http://', '')}{stream_path}/ws"
    
    async def _monitor_metrics(self) -> None:
        """Monitor stream metrics."""
        last_check = time.time()
        last_frames = 0
        last_bytes = 0
        
        try:
            while self.info.state == StreamState.ACTIVE:
                await asyncio.sleep(2)  # Update every 2 seconds
                
                if not self._recorder:
                    continue
                
                # Get current metrics from recorder
                metadata = self._recorder.metadata
                current_time = time.time()
                elapsed = current_time - last_check
                
                if elapsed > 0:
                    # Calculate rates
                    frame_delta = metadata.total_frames - last_frames
                    self.info.metrics.current_fps = frame_delta / elapsed
                    
                    # Update metrics
                    self.info.metrics.frames_sent = metadata.total_frames
                    self.info.metrics.last_update = current_time
                    
                    # Update averages
                    total_time = current_time - self.info.started_at
                    if total_time > 0:
                        self.info.metrics.average_bitrate = (
                            self.info.metrics.bytes_sent * 8 / total_time
                        )
                    
                    last_check = current_time
                    last_frames = metadata.total_frames
                
        except asyncio.CancelledError:
            logger.info(f"Metrics monitoring stopped for stream {self.info.stream_id}")
    
    async def _monitor_health(self) -> None:
        """Monitor stream health."""
        try:
            while self.info.state == StreamState.ACTIVE:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check if recording is still active
                if self._recorder and not self._recorder.is_recording:
                    self.info.health = StreamHealth.UNHEALTHY
                    logger.warning(f"Stream unhealthy: recorder not active {self.info.stream_id}")
                    continue
                
                # Check FPS
                if self.info.metrics.current_fps < self.config.frame_rate * 0.5:
                    self.info.health = StreamHealth.DEGRADED
                    logger.warning(f"Stream degraded: low FPS {self.info.stream_id}")
                    continue
                
                # Check buffer health
                if self.info.metrics.buffer_health < 50:
                    self.info.health = StreamHealth.DEGRADED
                    logger.warning(f"Stream degraded: low buffer {self.info.stream_id}")
                    continue
                
                # All checks passed
                self.info.health = StreamHealth.HEALTHY
                
        except asyncio.CancelledError:
            logger.info(f"Health monitoring stopped for stream {self.info.stream_id}")
    
    async def _notify_viewers_stream_ended(self) -> None:
        """Notify all viewers that stream has ended."""
        for viewer_queue in self._viewer_connections.values():
            try:
                await viewer_queue.put({"type": "stream_ended", "stream_id": self.info.stream_id})
            except Exception as e:
                logger.error(f"Failed to notify viewer: {e}")
    
    async def get_playlist_content(self) -> Optional[str]:
        """Get HLS playlist content.
        
        Returns:
            Playlist content if HLS stream
        """
        if self.config.protocol != StreamingProtocol.HLS:
            return None
        
        playlist_path = self.stream_dir / "playlist.m3u8"
        if not playlist_path.exists():
            return None
        
        try:
            return await asyncio.to_thread(playlist_path.read_text)
        except Exception as e:
            logger.error(f"Failed to read playlist: {e}")
            return None
    
    async def get_segment(self, segment_name: str) -> Optional[bytes]:
        """Get HLS segment data.
        
        Args:
            segment_name: Segment file name
            
        Returns:
            Segment data if exists
        """
        segment_path = self.stream_dir / segment_name
        if not segment_path.exists():
            return None
        
        try:
            return await asyncio.to_thread(segment_path.read_bytes)
        except Exception as e:
            logger.error(f"Failed to read segment: {e}")
            return None


class StreamingManager:
    """Manager for multiple streaming sessions.
    
    Coordinates streaming across the system:
    - Creates and manages streaming sessions
    - Tracks active streams
    - Handles viewer connections
    - Provides stream discovery
    - Enforces resource limits
    """
    
    def __init__(
        self,
        output_dir: str = "./streams",
        base_url: Optional[str] = None,
        max_concurrent_streams: int = 10,
    ) -> None:
        """Initialize streaming manager.
        
        Args:
            output_dir: Directory for stream output
            base_url: Base URL for accessing streams
            max_concurrent_streams: Maximum concurrent streams
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.max_concurrent_streams = max_concurrent_streams
        
        # Active streams
        self._streams: Dict[str, StreamingSession] = {}
        self._stream_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"StreamingManager initialized (max streams: {max_concurrent_streams})")
    
    async def start(self) -> None:
        """Start the streaming manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("StreamingManager started")
    
    async def stop(self) -> None:
        """Stop the streaming manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Stop all active streams
        async with self._stream_lock:
            for stream in list(self._streams.values()):
                try:
                    await stream.stop()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
            self._streams.clear()
        
        logger.info("StreamingManager stopped")
    
    async def create_stream(
        self,
        session_id: str,
        page: Page,
        config: StreamConfig,
    ) -> StreamInfo:
        """Create and start a new stream.
        
        Args:
            session_id: Browser session ID
            page: Playwright page to stream
            config: Stream configuration
            
        Returns:
            StreamInfo for the new stream
            
        Raises:
            ValueError: If too many concurrent streams
        """
        async with self._stream_lock:
            if len(self._streams) >= self.max_concurrent_streams:
                raise ValueError(
                    f"Maximum concurrent streams reached ({self.max_concurrent_streams})"
                )
            
            # Check if session already has a stream
            for stream in self._streams.values():
                if stream.session_id == session_id:
                    raise ValueError(f"Session {session_id} already has an active stream")
            
            # Create stream
            stream = StreamingSession(
                session_id=session_id,
                config=config,
                output_dir=str(self.output_dir),
                base_url=self.base_url,
            )
            
            # Start stream
            info = await stream.start(page)
            
            # Store stream
            self._streams[info.stream_id] = stream
            
            logger.info(f"Stream created: {info.stream_id} for session {session_id}")
            return info
    
    async def stop_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """Stop a stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Final StreamInfo if stream existed
        """
        async with self._stream_lock:
            stream = self._streams.get(stream_id)
            if not stream:
                return None
            
            info = await stream.stop()
            del self._streams[stream_id]
            
            return info
    
    async def get_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """Get stream information.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            StreamInfo if stream exists
        """
        stream = self._streams.get(stream_id)
        return stream.info if stream else None
    
    async def list_streams(
        self,
        session_id: Optional[str] = None,
    ) -> List[StreamInfo]:
        """List active streams.
        
        Args:
            session_id: Filter by session ID
            
        Returns:
            List of StreamInfo
        """
        streams = []
        async with self._stream_lock:
            for stream in self._streams.values():
                if session_id and stream.session_id != session_id:
                    continue
                streams.append(stream.info)
        
        return streams
    
    async def add_viewer(self, stream_id: str, viewer_id: str) -> bool:
        """Add a viewer to a stream.
        
        Args:
            stream_id: Stream identifier
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was added
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return False
        
        return await stream.add_viewer(viewer_id)
    
    async def remove_viewer(self, stream_id: str, viewer_id: str) -> bool:
        """Remove a viewer from a stream.
        
        Args:
            stream_id: Stream identifier
            viewer_id: Viewer identifier
            
        Returns:
            True if viewer was removed
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return False
        
        return await stream.remove_viewer(viewer_id)
    
    async def get_playlist(self, stream_id: str) -> Optional[str]:
        """Get HLS playlist for a stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Playlist content
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        
        return await stream.get_playlist_content()
    
    async def get_segment(self, stream_id: str, segment_name: str) -> Optional[bytes]:
        """Get HLS segment data.
        
        Args:
            stream_id: Stream identifier
            segment_name: Segment file name
            
        Returns:
            Segment data
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        
        return await stream.get_segment(segment_name)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old streams."""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._stream_lock:
                    to_remove = []
                    
                    for stream_id, stream in self._streams.items():
                        # Remove stopped streams
                        if stream.info.state == StreamState.STOPPED:
                            to_remove.append(stream_id)
                            continue
                        
                        # Check max duration
                        uptime = time.time() - stream.info.started_at
                        if uptime > stream.config.max_duration_seconds:
                            logger.info(f"Stream exceeded max duration: {stream_id}")
                            await stream.stop()
                            to_remove.append(stream_id)
                            continue
                        
                        # Remove streams with no viewers for too long
                        if (len(stream.info.viewer_ids) == 0 and
                            uptime > 300):  # 5 minutes with no viewers
                            logger.info(f"Stream abandoned: {stream_id}")
                            await stream.stop()
                            to_remove.append(stream_id)
                    
                    for stream_id in to_remove:
                        del self._streams[stream_id]
                    
                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} streams")
                        
        except asyncio.CancelledError:
            logger.info("Cleanup loop stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics.
        
        Returns:
            Dictionary with streaming stats
        """
        active_streams = sum(
            1 for s in self._streams.values()
            if s.info.state == StreamState.ACTIVE
        )
        
        total_viewers = sum(
            len(s.info.viewer_ids)
            for s in self._streams.values()
        )
        
        total_bandwidth = sum(
            s.info.metrics.current_bitrate
            for s in self._streams.values()
        )
        
        return {
            "total_streams": len(self._streams),
            "active_streams": active_streams,
            "total_viewers": total_viewers,
            "total_bandwidth_bps": total_bandwidth,
            "max_concurrent_streams": self.max_concurrent_streams,
            "utilization": len(self._streams) / self.max_concurrent_streams,
        }
