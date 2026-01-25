/**
 * FlyBrowser Completion Page JavaScript
 * Matrix rain animation with state-aware colors and copy functionality
 */

class MatrixRain {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.characters = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        this.fontSize = 14;
        this.columns = 0;
        this.drops = [];
        this.animationId = null;
        
        // Determine color based on success/failure state
        this.isSuccess = document.body.classList.contains('success');
        this.primaryColor = this.isSuccess ? '#00ff41' : '#ff4444';
        
        this.init();
        this.animate();
        
        // Handle resize
        window.addEventListener('resize', () => this.init());
    }
    
    init() {
        // Set canvas size to window size
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Calculate number of columns
        this.columns = Math.floor(this.canvas.width / this.fontSize);
        
        // Initialize drops (one per column, starting at random y)
        this.drops = [];
        for (let i = 0; i < this.columns; i++) {
            this.drops[i] = Math.random() * -100;
        }
    }
    
    draw() {
        // Semi-transparent black to create fade effect
        this.ctx.fillStyle = 'rgba(5, 5, 5, 0.05)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set font
        this.ctx.font = `${this.fontSize}px JetBrains Mono, monospace`;
        
        for (let i = 0; i < this.drops.length; i++) {
            // Random character
            const char = this.characters[Math.floor(Math.random() * this.characters.length)];
            
            // Calculate positions
            const x = i * this.fontSize;
            const y = this.drops[i] * this.fontSize;
            
            // Vary the opacity based on position for depth effect
            const opacity = Math.random() * 0.5 + 0.5;
            
            // Use state-aware color
            if (this.isSuccess) {
                this.ctx.fillStyle = `rgba(0, 255, 65, ${opacity})`;
            } else {
                this.ctx.fillStyle = `rgba(255, 68, 68, ${opacity})`;
            }
            
            // Draw character
            this.ctx.fillText(char, x, y);
            
            // Randomly reset drop to top with varying probability
            if (y > this.canvas.height && Math.random() > 0.975) {
                this.drops[i] = 0;
            }
            
            // Move drop down
            this.drops[i] += 0.5 + Math.random() * 0.5;
        }
    }
    
    animate() {
        this.draw();
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}

/**
 * Copy result data to clipboard
 */
function copyResult() {
    const resultElement = document.getElementById('result-data');
    if (!resultElement) return;
    
    const text = resultElement.textContent;
    const copyBtn = document.querySelector('.copy-btn');
    const copyText = copyBtn?.querySelector('.copy-text');
    
    navigator.clipboard.writeText(text).then(() => {
        // Visual feedback
        if (copyBtn) {
            copyBtn.classList.add('copied');
            if (copyText) {
                copyText.textContent = 'Copied!';
            }
            
            // Reset after 2 seconds
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                if (copyText) {
                    copyText.textContent = 'Copy';
                }
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy: ', err);
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            if (copyBtn) {
                copyBtn.classList.add('copied');
                if (copyText) {
                    copyText.textContent = 'Copied!';
                }
                setTimeout(() => {
                    copyBtn.classList.remove('copied');
                    if (copyText) {
                        copyText.textContent = 'Copy';
                    }
                }, 2000);
            }
        } catch (e) {
            console.error('Fallback copy failed: ', e);
        }
        document.body.removeChild(textarea);
    });
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Start matrix rain
    const matrixRain = new MatrixRain('matrix-rain');
    
    // Optional: Stop animation when page is hidden (battery saving)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            matrixRain.stop();
        } else {
            matrixRain.animate();
        }
    });
});
