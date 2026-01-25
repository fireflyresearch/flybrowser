/**
 * FlyBrowser Stream Player
 * Handles HLS and DASH playback with automatic retry logic
 */

class FlyBrowserPlayer {
    constructor(config) {
        this.config = {
            protocol: config.protocol || 'hls',
            hlsUrl: config.hlsUrl || '',
            dashUrl: config.dashUrl || '',
            qualityProfile: config.qualityProfile || '',
            maxRetries: config.maxRetries || 30,
            retryDelay: config.retryDelay || 2000,
        };
        
        // Determine if using LOCAL profile for ultra-low latency
        this.isLocalProfile = this.config.qualityProfile.startsWith('local_') || 
                              this.config.qualityProfile === 'studio';

        this.elements = {
            video: document.getElementById('video'),
            statusBadge: document.getElementById('status-badge'),
            statusText: document.getElementById('status-text'),
            loadingOverlay: document.getElementById('loading-overlay'),
            retryCount: document.getElementById('retry-count'),
            latency: document.getElementById('latency'),
            buffer: document.getElementById('buffer'),
            startTime: document.getElementById('start-time'),
            streamUrl: document.getElementById('stream-url'),
            copyBtn: document.getElementById('copy-btn'),
        };

        this.hls = null;
        this.dashPlayer = null;
        this.retryCount = 0;
        this.statsInterval = null;

        this.init();
    }

    init() {
        // Set start time
        if (this.elements.startTime) {
            this.elements.startTime.textContent = new Date().toLocaleTimeString();
        }

        // Set stream URL display
        if (this.elements.streamUrl) {
            this.elements.streamUrl.textContent = this.config.hlsUrl || this.config.dashUrl;
        }

        // Initialize player based on protocol
        this.initPlayer();

        // Setup copy button
        if (this.elements.copyBtn) {
            this.elements.copyBtn.addEventListener('click', () => this.copyUrl());
        }
    }

    setStatus(state, text) {
        if (this.elements.statusBadge) {
            this.elements.statusBadge.className = 'stream-badge ' + state;
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = text;
        }
        if (state === 'live' && this.elements.loadingOverlay) {
            this.elements.loadingOverlay.classList.add('hidden');
        }
    }

    updateStats() {
        if (this.hls && this.hls.media) {
            // Update latency
            const latency = this.hls.latency;
            if (latency !== undefined && latency > 0 && this.elements.latency) {
                this.elements.latency.textContent = latency.toFixed(1) + 's';
            }

            // Update buffer
            const buffered = this.elements.video.buffered;
            if (buffered.length > 0 && this.elements.buffer) {
                const bufferEnd = buffered.end(buffered.length - 1);
                const bufferLen = bufferEnd - this.elements.video.currentTime;
                this.elements.buffer.textContent = bufferLen.toFixed(1) + 's';
            }
        }
    }

    copyUrl() {
        const url = this.config.hlsUrl || this.config.dashUrl;
        navigator.clipboard.writeText(url).then(() => {
            if (this.elements.copyBtn) {
                this.elements.copyBtn.textContent = 'Copied!';
                this.elements.copyBtn.classList.add('copied');
                setTimeout(() => {
                    this.elements.copyBtn.textContent = 'Copy URL';
                    this.elements.copyBtn.classList.remove('copied');
                }, 2000);
            }
        });
    }

    initPlayer() {
        if (this.config.protocol === 'hls' && this.config.hlsUrl) {
            this.initHlsPlayer();
        } else if (this.config.protocol === 'dash' && this.config.dashUrl) {
            this.initDashPlayer();
        } else {
            this.setStatus('error', 'No stream URL');
        }
    }

    initHlsPlayer() {
        if (typeof Hls === 'undefined') {
            this.setStatus('error', 'HLS.js not loaded');
            return;
        }

        if (Hls.isSupported()) {
            this.initHlsJs();
        } else if (this.elements.video.canPlayType('application/vnd.apple.mpegurl')) {
            // Native HLS support (Safari)
            this.initNativeHls();
        } else {
            this.setStatus('error', 'HLS not supported');
        }
    }

    initHlsJs() {
        if (this.hls) {
            this.hls.destroy();
        }

        console.log('[FlyBrowser] Initializing HLS.js player');
        console.log('[FlyBrowser] HLS URL:', this.config.hlsUrl);
        console.log('[FlyBrowser] Quality Profile:', this.config.qualityProfile);
        console.log('[FlyBrowser] Low Latency Mode:', this.isLocalProfile);
        
        // Configure HLS.js based on quality profile
        const hlsConfig = this.isLocalProfile ? {
            // LOCAL PROFILES: Ultra-low latency settings
            enableWorker: true,
            lowLatencyMode: true,
            // Minimal buffering for sub-second latency
            backBufferLength: 10,
            maxBufferLength: 2,             // Only 2s buffer
            maxMaxBufferLength: 4,
            maxBufferSize: 0,
            maxBufferHole: 0.1,
            // Stay as close to live edge as possible
            liveSyncDurationCount: 1,       // 1 segment behind (0.5s)
            liveMaxLatencyDurationCount: 3, // Max 3 segments (1.5s)
            liveBackBufferLength: 10,
            liveDurationInfinity: true,
            liveMaxLatencyDuration: 3,
            liveSyncDuration: 0.5,
            // Fast checks and recovery
            highBufferWatchdogPeriod: 1,
            nudgeOffset: 0.05,
            nudgeMaxRetry: 3,
            startFragPrefetch: true,
            testBandwidth: false,
            // Very fast retries for localhost
            manifestLoadingMaxRetry: 20,
            manifestLoadingRetryDelay: 200,
            levelLoadingMaxRetry: 10,
            levelLoadingRetryDelay: 200,
            fragLoadingMaxRetry: 6,
            fragLoadingRetryDelay: 100,
            startLevel: -1,
            maxStarvationDelay: 1,
            maxLoadingDelay: 2,
            debug: false,
        } : {
            // STANDARD PROFILES: Balanced for internet streaming
            enableWorker: true,
            lowLatencyMode: false,
            // More buffering for smooth playback over internet
            backBufferLength: 30,
            maxBufferLength: 10,            // 10s buffer
            maxMaxBufferLength: 30,
            maxBufferSize: 0,
            maxBufferHole: 0.5,
            // Stay 3 segments behind for stability
            liveSyncDurationCount: 3,       // 3 segments (6s with 2s segments)
            liveMaxLatencyDurationCount: 6,
            liveBackBufferLength: 30,
            liveDurationInfinity: true,
            // Standard playback checks
            highBufferWatchdogPeriod: 2,
            nudgeOffset: 0.1,
            nudgeMaxRetry: 5,
            startFragPrefetch: true,
            testBandwidth: false,
            // Standard retry timing
            manifestLoadingMaxRetry: 20,
            manifestLoadingRetryDelay: 1000,
            levelLoadingMaxRetry: 10,
            levelLoadingRetryDelay: 1000,
            fragLoadingMaxRetry: 6,
            fragLoadingRetryDelay: 1000,
            startLevel: -1,
            maxStarvationDelay: 4,
            maxLoadingDelay: 4,
            debug: false,
        };
        
        this.hls = new Hls(hlsConfig);

        this.hls.loadSource(this.config.hlsUrl);
        this.hls.attachMedia(this.elements.video);

        this.hls.on(Hls.Events.MANIFEST_PARSED, (event, data) => {
            console.log('[FlyBrowser] Manifest parsed, levels:', data.levels.length);
            this.setStatus('live', 'Live');
            this.elements.video.play().catch((e) => {
                console.warn('[FlyBrowser] Autoplay blocked:', e.message);
            });
            this.retryCount = 0;

            // Start stats update interval
            if (this.statsInterval) {
                clearInterval(this.statsInterval);
            }
            this.statsInterval = setInterval(() => this.updateStats(), 1000);
        });

        this.hls.on(Hls.Events.MANIFEST_LOADING, () => {
            console.log('[FlyBrowser] Loading manifest...');
        });

        this.hls.on(Hls.Events.LEVEL_LOADED, (event, data) => {
            console.log('[FlyBrowser] Level loaded, fragments:', data.details.fragments.length);
            // Store live edge for sync checking
            this._liveEdge = data.details.live ? data.details.edge : null;
        });
        
        // Handle buffer stalling - seek to live edge
        this.hls.on(Hls.Events.FRAG_BUFFERED, () => {
            // Reset stall counter on successful buffer
            this._stallCount = 0;
        });
        
        // Handle playback stalls - but don't be too aggressive
        this.elements.video.addEventListener('waiting', () => {
            this._stallCount = (this._stallCount || 0) + 1;
            // Only log occasionally to reduce spam
            if (this._stallCount % 5 === 1) {
                console.log('[FlyBrowser] Playback buffering...');
            }
            
            // Only seek to live if stalled many times (10+) to avoid disrupting normal buffering
            if (this._stallCount >= 10) {
                console.log('[FlyBrowser] Excessive buffering, seeking to live edge');
                this.seekToLive();
                this._stallCount = 0;
            }
        });
        
        // Handle playback ending unexpectedly (live stream shouldn't end)
        this.elements.video.addEventListener('ended', () => {
            console.log('[FlyBrowser] Unexpected end, seeking to live');
            this.seekToLive();
        });

        this.hls.on(Hls.Events.ERROR, (event, data) => {
            console.error('[FlyBrowser] HLS Error:', data.type, data.details, data.fatal ? '(FATAL)' : '');
            if (data.fatal) {
                this.handleHlsError(data);
            } else if (data.details === Hls.ErrorDetails.BUFFER_STALLED_ERROR) {
                // Non-fatal buffer stall - try seeking to live
                console.warn('[FlyBrowser] Buffer stalled, seeking to live edge');
                this.seekToLive();
            }
        });
    }
    
    seekToLive() {
        if (!this.hls || !this.elements.video) return;
        
        try {
            // Get the live edge from HLS.js
            const liveEdge = this.hls.liveSyncPosition;
            if (liveEdge && liveEdge > 0) {
                const currentTime = this.elements.video.currentTime;
                const lag = liveEdge - currentTime;
                
                // Threshold and buffer depend on profile
                const threshold = this.isLocalProfile ? 2 : 15;  // 2s for local, 15s for internet
                const buffer = this.isLocalProfile ? 0.5 : 3;    // 0.5s for local, 3s for internet
                
                if (lag > threshold) {
                    console.log(`[FlyBrowser] Seeking to live edge: ${currentTime.toFixed(1)}s -> ${liveEdge.toFixed(1)}s (${lag.toFixed(1)}s behind)`);
                    this.elements.video.currentTime = liveEdge - buffer;
                    this.elements.video.play().catch(() => {});
                }
            }
        } catch (e) {
            console.warn('[FlyBrowser] Error seeking to live:', e);
        }
    }
    

    handleHlsError(data) {
        const errorType = data.type;
        const errorDetails = data.details;
        
        // Network errors - retry loading
        if (errorType === Hls.ErrorTypes.NETWORK_ERROR) {
            this.retryCount++;
            if (this.retryCount <= this.config.maxRetries) {
                if (this.elements.retryCount) {
                    this.elements.retryCount.textContent = `Retry ${this.retryCount}/${this.config.maxRetries}...`;
                }
                this.setStatus('connecting', 'Reconnecting...');
                setTimeout(() => {
                    this.hls.loadSource(this.config.hlsUrl);
                    this.hls.startLoad();
                }, this.config.retryDelay);
            } else {
                this.setStatus('error', 'Connection lost');
                if (this.elements.retryCount) {
                    this.elements.retryCount.textContent = 'Stream ended or unavailable. Refresh to reconnect.';
                }
            }
        } 
        // Media errors - try to recover
        else if (errorType === Hls.ErrorTypes.MEDIA_ERROR) {
            console.warn('[FlyBrowser] Media error, attempting recovery:', errorDetails);
            this.retryCount++;
            if (this.retryCount <= this.config.maxRetries) {
                if (this.elements.retryCount) {
                    this.elements.retryCount.textContent = `Recovering... (${this.retryCount}/${this.config.maxRetries})`;
                }
                // Try to recover from media error
                this.hls.recoverMediaError();
            } else {
                // If recovery fails multiple times, try full reload
                console.warn('[FlyBrowser] Recovery failed, reloading stream');
                this.retryCount = 0;
                this.hls.destroy();
                setTimeout(() => this.initHlsJs(), this.config.retryDelay);
            }
        }
        // Other fatal errors
        else {
            console.error('[FlyBrowser] Fatal error:', errorType, errorDetails);
            this.setStatus('error', 'Stream error');
            if (this.elements.retryCount) {
                this.elements.retryCount.textContent = 'Playback error. Refresh to try again.';
            }
        }
    }

    initNativeHls() {
        this.elements.video.src = this.config.hlsUrl;
        this.elements.video.addEventListener('loadedmetadata', () => {
            this.setStatus('live', 'Live');
            if (this.elements.loadingOverlay) {
                this.elements.loadingOverlay.classList.add('hidden');
            }
            this.elements.video.play().catch(() => {});
        });
        this.elements.video.addEventListener('error', () => {
            this.setStatus('error', 'Playback error');
        });
    }

    initDashPlayer() {
        if (typeof dashjs === 'undefined') {
            this.setStatus('error', 'dash.js not loaded');
            return;
        }

        this.dashPlayer = dashjs.MediaPlayer().create();
        this.dashPlayer.initialize(this.elements.video, this.config.dashUrl, true);
        this.dashPlayer.updateSettings({
            streaming: {
                lowLatencyEnabled: true,
                liveDelay: 2,
                liveCatchup: {
                    enabled: true
                }
            }
        });

        this.dashPlayer.on(dashjs.MediaPlayer.events.STREAM_INITIALIZED, () => {
            this.setStatus('live', 'Live');
            if (this.elements.loadingOverlay) {
                this.elements.loadingOverlay.classList.add('hidden');
            }
        });

        this.dashPlayer.on(dashjs.MediaPlayer.events.ERROR, () => {
            this.setStatus('error', 'Error');
        });
    }

    destroy() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
        }
        if (this.liveSyncInterval) {
            clearInterval(this.liveSyncInterval);
        }
        if (this.hls) {
            this.hls.destroy();
        }
        if (this.dashPlayer) {
            this.dashPlayer.reset();
        }
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.FlyBrowserPlayer = FlyBrowserPlayer;
}
