/**
 * FlyBrowser Completion Page JavaScript
 * Matrix rain animation, JSON tree explorer, and interactive features
 * Two-column layout with auto-expanded result data
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
        this.resizeTimeout = null;
        
        // Determine color based on success/failure state
        this.isSuccess = document.body.classList.contains('success');
        this.primaryColor = this.isSuccess ? '#00ff41' : '#ff4444';
        
        this.init();
        this.animate();
        
        // Handle resize with debounce for performance
        window.addEventListener('resize', () => this.handleResize());
        
        // Also handle orientation change on mobile
        window.addEventListener('orientationchange', () => {
            setTimeout(() => this.init(), 100);
        });
    }
    
    handleResize() {
        // Debounce resize events
        if (this.resizeTimeout) {
            clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = setTimeout(() => this.init(), 150);
    }
    
    init() {
        // Get actual viewport dimensions (handles mobile browsers with dynamic toolbars)
        const width = window.innerWidth || document.documentElement.clientWidth;
        const height = window.innerHeight || document.documentElement.clientHeight;
        
        // Set canvas size using device pixel ratio for crisp rendering on retina
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = width * dpr;
        this.canvas.height = height * dpr;
        
        // Scale canvas back down with CSS
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';
        
        // Scale context to account for device pixel ratio
        this.ctx.scale(dpr, dpr);
        
        // Calculate number of columns based on CSS size
        this.columns = Math.floor(width / this.fontSize);
        
        // Initialize drops (one per column, starting at random y)
        this.drops = [];
        for (let i = 0; i < this.columns; i++) {
            this.drops[i] = Math.random() * -100;
        }
        
        // Store dimensions for draw function
        this.width = width;
        this.height = height;
    }
    
    draw() {
        // Semi-transparent black to create fade effect
        this.ctx.fillStyle = 'rgba(5, 5, 5, 0.05)';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
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
            if (y > this.height && Math.random() > 0.975) {
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
 * JSON Tree Renderer - Creates an interactive, collapsible JSON tree view
 * Default behavior: fully expanded for result visibility
 */
class JsonTreeRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            defaultExpanded: true,      // Expand all by default
            maxAutoExpandDepth: 10,     // Max depth to auto-expand
            truncateStringsAt: 500,     // Truncate long strings
            ...options
        };
    }
    
    render(data) {
        this.container.innerHTML = '';
        const tree = document.createElement('div');
        tree.className = 'json-tree';
        tree.appendChild(this.createNode(data, '', 0));
        this.container.appendChild(tree);
    }
    
    createNode(value, key, depth) {
        const wrapper = document.createElement('div');
        wrapper.className = 'json-node';
        
        if (value === null) {
            wrapper.innerHTML = this.renderKeyValue(key, '<span class="json-null">null</span>', depth);
            return wrapper;
        }
        
        if (typeof value === 'string') {
            const escaped = this.escapeHtml(value);
            const maxLen = this.options.truncateStringsAt;
            const truncated = escaped.length > maxLen ? escaped.substring(0, maxLen) + '...' : escaped;
            wrapper.innerHTML = this.renderKeyValue(key, `<span class="json-string">"${truncated}"</span>`, depth);
            if (escaped.length > maxLen) {
                wrapper.title = value;
            }
            return wrapper;
        }
        
        if (typeof value === 'number') {
            wrapper.innerHTML = this.renderKeyValue(key, `<span class="json-number">${value}</span>`, depth);
            return wrapper;
        }
        
        if (typeof value === 'boolean') {
            wrapper.innerHTML = this.renderKeyValue(key, `<span class="json-boolean">${value}</span>`, depth);
            return wrapper;
        }
        
        // Arrays and Objects
        const isArray = Array.isArray(value);
        const entries = isArray ? value : Object.entries(value);
        const length = isArray ? value.length : Object.keys(value).length;
        const bracket = isArray ? ['[', ']'] : ['{', '}'];
        
        if (length === 0) {
            wrapper.innerHTML = this.renderKeyValue(key, `<span class="json-bracket">${bracket[0]}${bracket[1]}</span>`, depth);
            return wrapper;
        }
        
        // Collapsible container
        const container = document.createElement('div');
        container.className = 'json-collapsible';
        
        // Toggle header
        const header = document.createElement('div');
        header.className = 'json-toggle';
        header.innerHTML = `
            <span class="json-toggle-icon">▼</span>
            ${key ? `<span class="json-key">"${key}"</span><span class="json-colon">: </span>` : ''}
            <span class="json-bracket">${bracket[0]}</span>
            <span class="json-preview"> ${isArray ? `${length} items` : `${length} keys`} </span>
        `;
        header.onclick = (e) => {
            e.stopPropagation();
            this.toggleNode(container);
        };
        container.appendChild(header);
        
        // Children
        const children = document.createElement('div');
        children.className = 'json-children';
        
        if (isArray) {
            entries.forEach((item, index) => {
                const itemNode = document.createElement('div');
                itemNode.className = 'json-item';
                itemNode.appendChild(this.createNode(item, index.toString(), depth + 1));
                children.appendChild(itemNode);
            });
        } else {
            entries.forEach(([k, v]) => {
                const itemNode = document.createElement('div');
                itemNode.className = 'json-item';
                itemNode.appendChild(this.createNode(v, k, depth + 1));
                children.appendChild(itemNode);
            });
        }
        
        container.appendChild(children);
        
        // Closing bracket
        const closeBracket = document.createElement('div');
        closeBracket.innerHTML = `<span class="json-bracket">${bracket[1]}</span>`;
        closeBracket.style.paddingLeft = '0';
        container.appendChild(closeBracket);
        
        wrapper.appendChild(container);
        
        // Auto-collapse based on settings (default is expanded)
        const shouldCollapse = !this.options.defaultExpanded || depth >= this.options.maxAutoExpandDepth;
        if (shouldCollapse) {
            container.classList.add('json-collapsed');
        }
        
        return wrapper;
    }
    
    renderKeyValue(key, value, depth) {
        if (key) {
            return `<span class="json-key">"${key}"</span><span class="json-colon">: </span>${value}`;
        }
        return value;
    }
    
    toggleNode(container) {
        container.classList.toggle('json-collapsed');
    }
    
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    expandAll() {
        this.container.querySelectorAll('.json-collapsed').forEach(el => {
            el.classList.remove('json-collapsed');
        });
    }
    
    collapseAll() {
        this.container.querySelectorAll('.json-collapsible').forEach(el => {
            el.classList.add('json-collapsed');
        });
    }
}

// Global reference to tree renderer
let jsonTreeRenderer = null;
let currentResultView = 'tree';

/**
 * Toggle expandable sections
 */
function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    if (!content) return;
    
    const section = content.closest('.details-section');
    if (section) {
        section.classList.toggle('expanded');
    }
}

/**
 * Toggle task card expansion
 */
function toggleTaskExpand(element) {
    if (element) {
        element.classList.toggle('expanded');
    }
}

/**
 * Switch between tree and raw JSON view
 */
function setResultView(view) {
    currentResultView = view;
    
    const treeView = document.getElementById('result-tree');
    const rawView = document.getElementById('result-data');
    const btnTree = document.getElementById('btn-tree');
    const btnRaw = document.getElementById('btn-raw');
    
    if (view === 'tree') {
        treeView?.classList.remove('result-hidden');
        rawView?.classList.add('result-hidden');
        btnTree?.classList.add('active');
        btnRaw?.classList.remove('active');
    } else {
        treeView?.classList.add('result-hidden');
        rawView?.classList.remove('result-hidden');
        btnTree?.classList.remove('active');
        btnRaw?.classList.add('active');
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

/**
 * Expand all JSON nodes
 */
function expandAllJson() {
    if (jsonTreeRenderer) {
        jsonTreeRenderer.expandAll();
    }
}

/**
 * Collapse all JSON nodes
 */
function collapseAllJson() {
    if (jsonTreeRenderer) {
        jsonTreeRenderer.collapseAll();
    }
}

/**
 * Initialize JSON tree from embedded data
 * Result is fully expanded by default for better UX
 */
function initJsonTree() {
    const treeContainer = document.getElementById('result-tree');
    const jsonScript = document.getElementById('result-json');
    
    if (!treeContainer || !jsonScript) return;
    
    try {
        const jsonData = JSON.parse(jsonScript.textContent);
        if (jsonData !== null) {
            // Create tree renderer with expanded-by-default options
            jsonTreeRenderer = new JsonTreeRenderer(treeContainer, {
                defaultExpanded: true,
                maxAutoExpandDepth: 10,
                truncateStringsAt: 500
            });
            jsonTreeRenderer.render(jsonData);
        }
    } catch (e) {
        console.warn('Could not parse JSON for tree view:', e);
        // Fall back to raw view
        setResultView('raw');
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Start matrix rain
    const matrixRain = new MatrixRain('matrix-rain');
    
    // Initialize JSON tree (fully expanded)
    initJsonTree();
    
    // Optional: Stop animation when page is hidden (battery saving)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            matrixRain.stop();
        } else {
            matrixRain.animate();
        }
    });
});
