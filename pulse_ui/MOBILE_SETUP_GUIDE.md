# Mobile PWA Setup Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [Google Pay Integration](#google-pay-integration)
- [PWA Features Configuration](#pwa-features-configuration)
- [Testing & Debugging](#testing--debugging)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Deployment Checklist](#deployment-checklist)
- [Performance Optimization](#performance-optimization)
- [Support & Resources](#support--resources)

---

## Overview

This guide provides comprehensive instructions for setting up and deploying the Pulse UI mobile Progressive Web App (PWA). The PWA offers native-like functionality including offline support, push notifications, and payment integration.

### Key Features
- 📱 **Installable**: Add to home screen on iOS and Android
- 🔄 **Offline Support**: Service worker-based caching
- 💳 **Google Pay**: Integrated payment processing
- 🔔 **Push Notifications**: Real-time updates
- ⚡ **Fast Loading**: Optimized performance
- 🔒 **Secure**: HTTPS-only with modern security practices

---

## Prerequisites

### Required Software
- Node.js (v16.x or higher)
- npm or yarn package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)
- SSL certificate (for HTTPS)

### Development Tools
```bash
# Install required development dependencies
npm install --save-dev workbox-cli
npm install --save-dev lighthouse
npm install --save-dev pwa-asset-generator
```

### Required Accounts
- Google Developer Account (for Google Pay)
- Firebase Account (for push notifications - optional)
- SSL Certificate Authority (Let's Encrypt recommended)

---

## Installation Instructions

### Step 1: Project Setup

```bash
# Clone the repository
git clone https://github.com/jetgause/Library-.git
cd Library-/pulse_ui

# Install dependencies
npm install

# Create environment configuration
cp .env.example .env.local
```

### Step 2: Configure Environment Variables

Edit `.env.local` with your specific configuration:

```env
# App Configuration
VITE_APP_NAME=Pulse UI
VITE_APP_SHORT_NAME=Pulse
VITE_APP_DESCRIPTION=Library Management System
VITE_APP_VERSION=1.0.0

# API Configuration
VITE_API_URL=https://your-api-domain.com
VITE_API_TIMEOUT=30000

# Google Pay Configuration
VITE_GOOGLE_PAY_MERCHANT_ID=your_merchant_id
VITE_GOOGLE_PAY_MERCHANT_NAME=Your Merchant Name
VITE_GOOGLE_PAY_ENVIRONMENT=TEST  # or PRODUCTION

# Push Notifications (Firebase)
VITE_FIREBASE_API_KEY=your_firebase_api_key
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id

# Security
VITE_ENABLE_CSP=true
VITE_ENABLE_HSTS=true
```

### Step 3: Manifest Configuration

Update `public/manifest.json`:

```json
{
  "name": "Pulse UI - Library Management",
  "short_name": "Pulse",
  "description": "Modern library management system with mobile support",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#4F46E5",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "/icons/icon-72x72.png",
      "sizes": "72x72",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-96x96.png",
      "sizes": "96x96",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-128x128.png",
      "sizes": "128x128",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-384x384.png",
      "sizes": "384x384",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ],
  "screenshots": [
    {
      "src": "/screenshots/mobile-1.png",
      "sizes": "540x720",
      "type": "image/png",
      "form_factor": "narrow"
    },
    {
      "src": "/screenshots/desktop-1.png",
      "sizes": "1280x720",
      "type": "image/png",
      "form_factor": "wide"
    }
  ],
  "categories": ["education", "productivity"],
  "iarc_rating_id": "e84b072d-71b3-4d3e-86ae-31a8ce4e53b7"
}
```

### Step 4: Service Worker Setup

Create `public/service-worker.js`:

```javascript
import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { StaleWhileRevalidate, CacheFirst, NetworkFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';

// Precache all assets generated by your build process
precacheAndRoute(self.__WB_MANIFEST);

// Cache API responses
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new NetworkFirst({
    cacheName: 'api-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 5 * 60, // 5 minutes
      }),
    ],
  })
);

// Cache images
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'image-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
      }),
      new CacheableResponsePlugin({
        statuses: [0, 200],
      }),
    ],
  })
);

// Cache Google Fonts
registerRoute(
  ({ url }) => url.origin === 'https://fonts.googleapis.com' ||
               url.origin === 'https://fonts.gstatic.com',
  new StaleWhileRevalidate({
    cacheName: 'google-fonts',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 20,
        maxAgeSeconds: 365 * 24 * 60 * 60, // 1 year
      }),
    ],
  })
);

// Handle offline page
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match('/offline.html');
      })
    );
  }
});

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncData());
  }
});

async function syncData() {
  // Implement your data sync logic
  console.log('Syncing data in background...');
}

// Push notification handling
self.addEventListener('push', (event) => {
  const data = event.data?.json() ?? {};
  const title = data.title || 'Pulse UI Notification';
  const options = {
    body: data.body || 'You have a new notification',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [200, 100, 200],
    data: data.url || '/',
    actions: data.actions || [],
  };

  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow(event.notification.data)
  );
});
```

### Step 5: Build and Test Locally

```bash
# Development build with hot reload
npm run dev

# Production build
npm run build

# Preview production build locally
npm run preview

# Test PWA functionality
npm run test:pwa
```

---

## Google Pay Integration

### Step 1: Google Pay Setup

1. **Register for Google Pay API**
   - Visit [Google Pay Business Console](https://pay.google.com/business/console)
   - Create a new project
   - Enable the Google Pay API
   - Obtain your Merchant ID

2. **Configure Payment Gateway**
   - Set up your payment gateway (Stripe, Square, etc.)
   - Obtain gateway credentials
   - Configure webhook endpoints

### Step 2: Implementation

Create `src/utils/googlePay.js`:

```javascript
/**
 * Google Pay Configuration
 */
const baseRequest = {
  apiVersion: 2,
  apiVersionMinor: 0
};

const allowedCardNetworks = ["AMEX", "DISCOVER", "JCB", "MASTERCARD", "VISA"];
const allowedCardAuthMethods = ["PAN_ONLY", "CRYPTOGRAM_3DS"];

const tokenizationSpecification = {
  type: 'PAYMENT_GATEWAY',
  parameters: {
    gateway: 'example', // Replace with your gateway
    gatewayMerchantId: import.meta.env.VITE_GOOGLE_PAY_GATEWAY_MERCHANT_ID
  }
};

const baseCardPaymentMethod = {
  type: 'CARD',
  parameters: {
    allowedAuthMethods: allowedCardAuthMethods,
    allowedCardNetworks: allowedCardNetworks
  }
};

const cardPaymentMethod = Object.assign(
  {},
  baseCardPaymentMethod,
  {
    tokenizationSpecification: tokenizationSpecification
  }
);

export function getGoogleIsReadyToPayRequest() {
  return Object.assign(
    {},
    baseRequest,
    {
      allowedPaymentMethods: [baseCardPaymentMethod]
    }
  );
}

export function getGooglePaymentDataRequest(amount, currency = 'USD') {
  const paymentDataRequest = Object.assign({}, baseRequest);
  paymentDataRequest.allowedPaymentMethods = [cardPaymentMethod];
  paymentDataRequest.transactionInfo = {
    totalPriceStatus: 'FINAL',
    totalPrice: amount.toString(),
    currencyCode: currency,
    countryCode: 'US'
  };
  paymentDataRequest.merchantInfo = {
    merchantId: import.meta.env.VITE_GOOGLE_PAY_MERCHANT_ID,
    merchantName: import.meta.env.VITE_GOOGLE_PAY_MERCHANT_NAME
  };

  return paymentDataRequest;
}

export async function initGooglePay() {
  if (!window.google) {
    throw new Error('Google Pay API not loaded');
  }

  const paymentsClient = new google.payments.api.PaymentsClient({
    environment: import.meta.env.VITE_GOOGLE_PAY_ENVIRONMENT || 'TEST'
  });

  return paymentsClient;
}

export async function processPayment(paymentsClient, amount, currency) {
  try {
    const paymentDataRequest = getGooglePaymentDataRequest(amount, currency);
    const paymentData = await paymentsClient.loadPaymentData(paymentDataRequest);
    
    // Send payment token to your server
    return await processPaymentToken(paymentData.paymentMethodData.tokenizationData.token);
  } catch (error) {
    console.error('Payment failed:', error);
    throw error;
  }
}

async function processPaymentToken(token) {
  const response = await fetch('/api/payments/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ token })
  });

  if (!response.ok) {
    throw new Error('Payment processing failed');
  }

  return response.json();
}
```

### Step 3: Add Google Pay Button Component

Create `src/components/GooglePayButton.jsx`:

```jsx
import React, { useEffect, useState } from 'react';
import { initGooglePay, getGoogleIsReadyToPayRequest, processPayment } from '../utils/googlePay';

export default function GooglePayButton({ amount, currency = 'USD', onSuccess, onError }) {
  const [paymentsClient, setPaymentsClient] = useState(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    async function initialize() {
      try {
        const client = await initGooglePay();
        const isReadyToPayRequest = getGoogleIsReadyToPayRequest();
        const response = await client.isReadyToPay(isReadyToPayRequest);
        
        if (response.result) {
          setPaymentsClient(client);
          setIsReady(true);
        }
      } catch (error) {
        console.error('Google Pay initialization failed:', error);
        onError?.(error);
      }
    }

    initialize();
  }, []);

  const handleClick = async () => {
    if (!paymentsClient) return;

    try {
      const result = await processPayment(paymentsClient, amount, currency);
      onSuccess?.(result);
    } catch (error) {
      onError?.(error);
    }
  };

  if (!isReady) return null;

  return (
    <button
      onClick={handleClick}
      className="google-pay-button"
      aria-label="Pay with Google Pay"
    >
      <img 
        src="https://www.gstatic.com/instantbuy/svg/light/en.svg" 
        alt="Google Pay"
      />
    </button>
  );
}
```

### Step 4: Testing Google Pay

```bash
# Use test card numbers in TEST environment
# Visa: 4111111111111111
# Mastercard: 5555555555554444
# Amex: 378282246310005

# Test different scenarios
npm run test:google-pay
```

---

## PWA Features Configuration

### Install Prompt

Create `src/hooks/useInstallPrompt.js`:

```javascript
import { useState, useEffect } from 'react';

export function useInstallPrompt() {
  const [installPrompt, setInstallPrompt] = useState(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setInstallPrompt(e);
    };

    window.addEventListener('beforeinstallprompt', handler);

    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsInstalled(true);
    }

    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const promptInstall = async () => {
    if (!installPrompt) return false;

    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    
    if (outcome === 'accepted') {
      setIsInstalled(true);
      setInstallPrompt(null);
      return true;
    }
    
    return false;
  };

  return { promptInstall, canInstall: !!installPrompt, isInstalled };
}
```

### Push Notifications

Create `src/utils/notifications.js`:

```javascript
export async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    console.warn('Notifications not supported');
    return false;
  }

  if (Notification.permission === 'granted') {
    return true;
  }

  if (Notification.permission !== 'denied') {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }

  return false;
}

export async function subscribeToPushNotifications() {
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    console.warn('Push notifications not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(
        import.meta.env.VITE_VAPID_PUBLIC_KEY
      )
    });

    // Send subscription to server
    await fetch('/api/notifications/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(subscription)
    });

    return subscription;
  } catch (error) {
    console.error('Push subscription failed:', error);
    return null;
  }
}

function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/\-/g, '+')
    .replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}
```

### Offline Support

Create `public/offline.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Offline - Pulse UI</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      text-align: center;
      padding: 20px;
    }
    .container {
      max-width: 500px;
    }
    h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    p { font-size: 1.2rem; opacity: 0.9; margin-bottom: 2rem; }
    button {
      background: white;
      color: #667eea;
      border: none;
      padding: 12px 24px;
      font-size: 1rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { transform: scale(1.05); }
  </style>
</head>
<body>
  <div class="container">
    <h1>📱 You're Offline</h1>
    <p>No internet connection detected. Some features may be limited.</p>
    <button onclick="window.location.reload()">Try Again</button>
  </div>
</body>
</html>
```

---

## Testing & Debugging

### Lighthouse PWA Audit

```bash
# Run Lighthouse audit
npm run lighthouse

# Or use Chrome DevTools
# 1. Open Chrome DevTools (F12)
# 2. Go to Lighthouse tab
# 3. Select "Progressive Web App"
# 4. Click "Generate report"
```

### PWA Testing Checklist

- [ ] Manifest is properly configured
- [ ] Service worker is registered
- [ ] HTTPS is enabled
- [ ] Icons are all sizes present
- [ ] App is installable
- [ ] Offline page works
- [ ] Cache strategies are effective
- [ ] Push notifications work
- [ ] Google Pay integration functions
- [ ] Performance score > 90
- [ ] Accessibility score > 90

### Browser Testing

Test on multiple devices and browsers:

```bash
# iOS Safari
- Test on iPhone/iPad (iOS 12.2+)
- Verify Add to Home Screen

# Android Chrome
- Test on Android device (8.0+)
- Verify install banner

# Desktop Browsers
- Chrome (latest)
- Edge (latest)
- Firefox (latest)
```

### Debug Tools

```javascript
// Add to your main.js for debugging
if (import.meta.env.DEV) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.getRegistrations().then(registrations => {
      console.log('Service Workers:', registrations);
    });
  });
}
```

---

## Troubleshooting

### Common Issues

#### 1. Service Worker Not Updating

**Problem**: Changes not reflecting after deployment

**Solution**:
```javascript
// Add to service worker
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => caches.delete(cacheName))
      );
    })
  );
});
```

#### 2. App Not Installable

**Problem**: Install prompt doesn't appear

**Checklist**:
- ✅ Served over HTTPS
- ✅ Manifest file properly linked
- ✅ Service worker registered
- ✅ Icons include 192x192 and 512x512
- ✅ `start_url` is valid
- ✅ `display` is set correctly

#### 3. Google Pay Not Loading

**Problem**: Google Pay button doesn't appear

**Solutions**:
```javascript
// Check if Google Pay is loaded
if (typeof google === 'undefined' || !google.payments) {
  console.error('Google Pay script not loaded');
  // Load script manually
  const script = document.createElement('script');
  script.src = 'https://pay.google.com/gp/p/js/pay.js';
  script.async = true;
  document.head.appendChild(script);
}

// Verify environment configuration
console.log('Google Pay Config:', {
  merchantId: import.meta.env.VITE_GOOGLE_PAY_MERCHANT_ID,
  environment: import.meta.env.VITE_GOOGLE_PAY_ENVIRONMENT
});
```

#### 4. Push Notifications Not Working

**Problem**: Notifications not received

**Checklist**:
- ✅ Permission granted
- ✅ Service worker active
- ✅ VAPID keys configured
- ✅ Subscription sent to server
- ✅ Notification payload correct

**Debug**:
```javascript
// Test notification
navigator.serviceWorker.ready.then(registration => {
  registration.showNotification('Test', {
    body: 'Testing notifications',
    icon: '/icons/icon-192x192.png'
  });
});
```

#### 5. Offline Mode Not Working

**Problem**: App doesn't work offline

**Solution**:
```javascript
// Verify cache strategies
caches.keys().then(keys => console.log('Cache keys:', keys));

// Check cached resources
caches.open('your-cache-name').then(cache => {
  cache.keys().then(requests => {
    console.log('Cached requests:', requests.map(r => r.url));
  });
});
```

### iOS-Specific Issues

#### Add to Home Screen on iOS

Add to `index.html`:
```html
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Pulse">
<link rel="apple-touch-icon" href="/icons/icon-152x152.png">
<link rel="apple-touch-icon" sizes="180x180" href="/icons/icon-180x180.png">
```

---

## Security Considerations

### HTTPS Requirements

**Mandatory for PWA features**:
- Service Workers
- Push Notifications
- Google Pay
- Geolocation
- Camera/Microphone

**Setup SSL Certificate**:
```bash
# Using Let's Encrypt (recommended)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Content Security Policy (CSP)

Add to `index.html` or server headers:

```html
<meta http-equiv="Content-Security-Policy" content="
  default-src 'self';
  script-src 'self' 'unsafe-inline' https://pay.google.com;
  style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
  font-src 'self' https://fonts.gstatic.com;
  img-src 'self' data: https:;
  connect-src 'self' https://your-api.com;
  frame-src https://pay.google.com;
">
```

### CORS Configuration

Server-side CORS setup (Node.js/Express):

```javascript
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'https://yourdomain.com');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Access-Control-Allow-Credentials', 'true');
  next();
});
```

### Security Headers

Add to server configuration:

```nginx
# Nginx configuration
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "geolocation=(self), microphone=(), camera=()" always;
```

### Data Encryption

```javascript
// Encrypt sensitive data before storing
import CryptoJS from 'crypto-js';

export function encryptData(data, key) {
  return CryptoJS.AES.encrypt(JSON.stringify(data), key).toString();
}

export function decryptData(encryptedData, key) {
  const bytes = CryptoJS.AES.decrypt(encryptedData, key);
  return JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
}

// Usage
const encrypted = encryptData(userData, import.meta.env.VITE_ENCRYPTION_KEY);
localStorage.setItem('userData', encrypted);
```

### API Security

```javascript
// Implement API request signing
export async function signedFetch(url, options = {}) {
  const timestamp = Date.now();
  const nonce = Math.random().toString(36);
  
  const signature = await generateSignature({
    method: options.method || 'GET',
    url,
    timestamp,
    nonce,
    body: options.body
  });

  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'X-Timestamp': timestamp,
      'X-Nonce': nonce,
      'X-Signature': signature
    }
  });
}
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] **Environment Configuration**
  - [ ] Production environment variables set
  - [ ] API endpoints configured
  - [ ] Google Pay merchant ID updated
  - [ ] Firebase credentials configured

- [ ] **Build Optimization**
  - [ ] Run production build (`npm run build`)
  - [ ] Verify bundle size is optimized
  - [ ] Check for console errors
  - [ ] Test all features in production mode

- [ ] **Assets & Resources**
  - [ ] All icons generated (72px to 512px)
  - [ ] Screenshots added for app stores
  - [ ] Offline page created
  - [ ] Favicon configured

- [ ] **Security**
  - [ ] SSL certificate installed
  - [ ] CSP headers configured
  - [ ] Security headers added
  - [ ] CORS properly configured

### Testing Phase

- [ ] **Functional Testing**
  - [ ] All pages load correctly
  - [ ] Navigation works smoothly
  - [ ] Forms submit properly
  - [ ] API calls successful
  - [ ] Google Pay processes payments

- [ ] **PWA Testing**
  - [ ] Service worker registers
  - [ ] App installable on mobile
  - [ ] Offline mode works
  - [ ] Push notifications send/receive
  - [ ] Cache strategies effective

- [ ] **Cross-Browser Testing**
  - [ ] Chrome (desktop & mobile)
  - [ ] Safari (iOS & macOS)
  - [ ] Firefox
  - [ ] Edge
  - [ ] Samsung Internet

- [ ] **Performance Testing**
  - [ ] Lighthouse score > 90
  - [ ] First Contentful Paint < 2s
  - [ ] Time to Interactive < 5s
  - [ ] Total bundle size < 500KB

### Deployment Steps

1. **Build Production Assets**
```bash
npm run build
npm run preview  # Test production build locally
```

2. **Deploy to Server**
```bash
# Example: Deploy to Netlify
netlify deploy --prod

# Example: Deploy to Vercel
vercel --prod

# Example: Deploy to custom server
rsync -avz dist/ user@server:/var/www/pulse-ui/
```

3. **Configure Server**
```nginx
# Nginx configuration example
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    root /var/www/pulse-ui;
    index index.html;

    # Service worker cache control
    location /service-worker.js {
        add_header Cache-Control "no-cache";
        expires off;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

4. **Verify Deployment**
```bash
# Test HTTPS
curl -I https://yourdomain.com

# Test service worker
curl https://yourdomain.com/service-worker.js

# Test manifest
curl https://yourdomain.com/manifest.json
```

### Post-Deployment

- [ ] **Monitoring**
  - [ ] Set up error tracking (Sentry, LogRocket)
  - [ ] Configure analytics (Google Analytics, Plausible)
  - [ ] Monitor performance metrics
  - [ ] Set up uptime monitoring

- [ ] **User Testing**
  - [ ] Beta test with real users
  - [ ] Collect feedback
  - [ ] Monitor crash reports
  - [ ] Track user engagement

- [ ] **Documentation**
  - [ ] Update user documentation
  - [ ] Document deployment process
  - [ ] Create rollback plan
  - [ ] Update changelog

### Rollback Plan

```bash
# Keep previous build
mv dist dist.backup
npm run build

# If issues occur, restore previous version
rm -rf dist
mv dist.backup dist
```

---

## Performance Optimization

### Bundle Size Optimization

```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['@headlessui/react', '@heroicons/react'],
          utils: ['date-fns', 'lodash-es']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
};
```

### Image Optimization

```bash
# Install sharp for image processing
npm install --save-dev sharp

# Generate optimized images
npm run optimize:images
```

### Code Splitting

```javascript
// Lazy load routes
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Library = lazy(() => import('./pages/Library'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/library" element={<Library />} />
      </Routes>
    </Suspense>
  );
}
```

### Service Worker Optimization

```javascript
// Implement efficient caching strategies
const CACHE_VERSION = 'v1.0.0';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/offline.html',
  '/styles.css',
  '/main.js'
];

// Only precache essential resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then(cache => {
      return cache.addAll(PRECACHE_URLS);
    })
  );
});
```

---

## Support & Resources

### Official Documentation

- [MDN PWA Guide](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [Google Web Fundamentals](https://developers.google.com/web/fundamentals)
- [Google Pay Web Integration](https://developers.google.com/pay/api/web)
- [Web Push Notifications](https://web.dev/push-notifications/)

### Testing Tools

- [Lighthouse](https://developers.google.com/web/tools/lighthouse)
- [PWA Builder](https://www.pwabuilder.com/)
- [Web.dev Measure](https://web.dev/measure/)
- [Chrome DevTools](https://developer.chrome.com/docs/devtools/)

### Community Resources

- [PWA Slack Community](https://bit.ly/go-pwa-slack)
- [Stack Overflow - PWA Tag](https://stackoverflow.com/questions/tagged/progressive-web-apps)
- [Reddit r/PWA](https://www.reddit.com/r/PWA/)

### Getting Help

For issues specific to this project:

1. **Check Documentation**: Review this guide and related docs
2. **Search Issues**: Check [GitHub Issues](https://github.com/jetgause/Library-/issues)
3. **Create Issue**: Open a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs
   - Environment details

### Contact

- **Project Repository**: https://github.com/jetgause/Library-
- **Maintainer**: @jetgause
- **Email**: support@pulseui.com (if applicable)

---

## Appendix

### A. Environment Variables Reference

```env
# Complete list of environment variables

# App
VITE_APP_NAME=Pulse UI
VITE_APP_SHORT_NAME=Pulse
VITE_APP_DESCRIPTION=Library Management System
VITE_APP_VERSION=1.0.0
VITE_APP_THEME_COLOR=#4F46E5

# API
VITE_API_URL=https://api.yourdomain.com
VITE_API_TIMEOUT=30000
VITE_API_KEY=your_api_key

# Google Pay
VITE_GOOGLE_PAY_MERCHANT_ID=your_merchant_id
VITE_GOOGLE_PAY_MERCHANT_NAME=Your Business Name
VITE_GOOGLE_PAY_ENVIRONMENT=PRODUCTION
VITE_GOOGLE_PAY_GATEWAY=stripe
VITE_GOOGLE_PAY_GATEWAY_MERCHANT_ID=your_gateway_merchant_id

# Firebase
VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX
VITE_VAPID_PUBLIC_KEY=your_vapid_public_key

# Security
VITE_ENCRYPTION_KEY=your_encryption_key
VITE_ENABLE_CSP=true
VITE_ENABLE_HSTS=true

# Features
VITE_ENABLE_PUSH_NOTIFICATIONS=true
VITE_ENABLE_GOOGLE_PAY=true
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_ANALYTICS=true
```

### B. Browser Support Matrix

| Feature | Chrome | Firefox | Safari | Edge | Samsung Internet |
|---------|--------|---------|--------|------|------------------|
| Service Workers | 40+ | 44+ | 11.1+ | 17+ | 4.0+ |
| Web App Manifest | 39+ | ❌ | 11.3+ | 17+ | 4.0+ |
| Push Notifications | 42+ | 44+ | 16.0+ | 17+ | 4.0+ |
| Google Pay | Latest | Latest | Latest | Latest | Latest |
| IndexedDB | 24+ | 16+ | 10+ | 12+ | 4.0+ |
| Cache API | 40+ | 39+ | 11.1+ | 17+ | 4.0+ |

### C. Icon Sizes Reference

Required icon sizes for optimal PWA experience:

- 72x72 (Android)
- 96x96 (Android)
- 128x128 (Android, Chrome Web Store)
- 144x144 (Android)
- 152x152 (iOS)
- 180x180 (iOS)
- 192x192 (Android, required for installability)
- 384x384 (Android)
- 512x512 (Android, required for installability)

### D. Useful Commands

```bash
# Development
npm run dev              # Start dev server
npm run build            # Production build
npm run preview          # Preview production build
npm run lint             # Run linter
npm run format           # Format code

# Testing
npm run test             # Run tests
npm run test:watch       # Watch mode
npm run test:coverage    # Coverage report
npm run test:pwa         # PWA tests
npm run lighthouse       # Lighthouse audit

# Optimization
npm run analyze          # Bundle analysis
npm run optimize:images  # Optimize images
npm run generate:icons   # Generate PWA icons

# Deployment
npm run deploy:dev       # Deploy to dev
npm run deploy:staging   # Deploy to staging
npm run deploy:prod      # Deploy to production
```

---

**Last Updated**: 2025-12-10  
**Version**: 1.0.0  
**Maintainer**: @jetgause

For the latest updates and information, visit the [GitHub repository](https://github.com/jetgause/Library-).
