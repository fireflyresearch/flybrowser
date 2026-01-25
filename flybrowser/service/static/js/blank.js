/**
 * FlyBrowser Blank Page JavaScript
 * Matrix rain animation effect
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
        
        // Green text
        this.ctx.fillStyle = '#00ff41';
        this.ctx.font = `${this.fontSize}px JetBrains Mono, monospace`;
        
        for (let i = 0; i < this.drops.length; i++) {
            // Random character
            const char = this.characters[Math.floor(Math.random() * this.characters.length)];
            
            // Calculate x position
            const x = i * this.fontSize;
            const y = this.drops[i] * this.fontSize;
            
            // Vary the opacity based on position for depth effect
            const opacity = Math.random() * 0.5 + 0.5;
            this.ctx.fillStyle = `rgba(0, 255, 65, ${opacity})`;
            
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
