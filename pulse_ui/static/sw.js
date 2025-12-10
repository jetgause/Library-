const CACHE_NAME = 'pulse-v1.0.0';
const urlsToCache = [
  '/',
  '/static/css/custom.css',
  '/static/js/dashboard.js',
  '/tools',
  '/monitoring',
  '/settings'
];

// Install service worker and cache files
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

// Serve from cache when offline
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
      .catch(() => caches.match('/'))
  );
});