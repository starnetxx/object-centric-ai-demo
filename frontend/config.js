// Configuration for different environments
const config = {
    // For local development
    local: {
        apiUrl: 'http://localhost:8000'
    },
    // For production (Railway deployment)
    production: {
        apiUrl: 'https://web-production-e7ea7.up.railway.app'  // Your actual Railway URL
    },
    // For Netlify deployment
    netlify: {
        apiUrl: 'https://web-production-e7ea7.up.railway.app'  // Your actual Railway URL
    }
};

// Auto-detect environment
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const isNetlify = window.location.hostname.includes('netlify.app') || window.location.hostname.includes('netlify.com');

let currentConfig;
if (isLocal) {
    currentConfig = config.local;
} else if (isNetlify) {
    currentConfig = config.netlify;
} else {
    currentConfig = config.production;
}

// Export for use in index.html
window.API_CONFIG = currentConfig;
