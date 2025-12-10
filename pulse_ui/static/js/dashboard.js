/**
 * Dashboard Utilities
 * Manages WebSocket connections, API calls, metrics updates, and chart rendering
 */

// ============================================================================
// WebSocket Manager
// ============================================================================

class WebSocketManager {
    constructor(url, options = {}) {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = options.reconnectInterval || 3000;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
        this.reconnectAttempts = 0;
        this.handlers = {
            open: [],
            message: [],
            error: [],
            close: []
        };
        this.isManualClose = false;
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.handleReconnect();
        }
    }

    setupEventHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.handlers.open.forEach(handler => handler(event));
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handlers.message.forEach(handler => handler(data));
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            this.handlers.error.forEach(handler => handler(event));
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed');
            this.handlers.close.forEach(handler => handler(event));
            
            if (!this.isManualClose) {
                this.handleReconnect();
            }
        };
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
        }
    }

    on(event, handler) {
        if (this.handlers[event]) {
            this.handlers[event].push(handler);
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket is not open. Message not sent:', data);
        }
    }

    close() {
        this.isManualClose = true;
        if (this.ws) {
            this.ws.close();
        }
    }
}

// ============================================================================
// API Client
// ============================================================================

class APIClient {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // Dashboard-specific endpoints
    async getMetrics(timeRange = '1h') {
        return this.get(`/metrics?range=${timeRange}`);
    }

    async getSystemStatus() {
        return this.get('/system/status');
    }

    async getAlerts() {
        return this.get('/alerts');
    }

    async getServices() {
        return this.get('/services');
    }

    async getServiceDetails(serviceId) {
        return this.get(`/services/${serviceId}`);
    }
}

// ============================================================================
// Metrics Manager
// ============================================================================

class MetricsManager {
    constructor(apiClient) {
        this.api = apiClient;
        this.metrics = {
            cpu: [],
            memory: [],
            disk: [],
            network: [],
            latency: [],
            errors: []
        };
        this.maxDataPoints = 50;
    }

    addDataPoint(metric, timestamp, value) {
        if (!this.metrics[metric]) {
            this.metrics[metric] = [];
        }

        this.metrics[metric].push({ timestamp, value });

        // Keep only the last maxDataPoints
        if (this.metrics[metric].length > this.maxDataPoints) {
            this.metrics[metric].shift();
        }
    }

    getMetric(metric) {
        return this.metrics[metric] || [];
    }

    getAllMetrics() {
        return this.metrics;
    }

    clearMetric(metric) {
        if (this.metrics[metric]) {
            this.metrics[metric] = [];
        }
    }

    clearAll() {
        Object.keys(this.metrics).forEach(metric => {
            this.metrics[metric] = [];
        });
    }

    async fetchMetrics(timeRange = '1h') {
        try {
            const data = await this.api.getMetrics(timeRange);
            this.processMetricsData(data);
            return data;
        } catch (error) {
            console.error('Error fetching metrics:', error);
            throw error;
        }
    }

    processMetricsData(data) {
        if (data.cpu) {
            data.cpu.forEach(point => this.addDataPoint('cpu', point.timestamp, point.value));
        }
        if (data.memory) {
            data.memory.forEach(point => this.addDataPoint('memory', point.timestamp, point.value));
        }
        if (data.disk) {
            data.disk.forEach(point => this.addDataPoint('disk', point.timestamp, point.value));
        }
        if (data.network) {
            data.network.forEach(point => this.addDataPoint('network', point.timestamp, point.value));
        }
    }

    calculateAverage(metric) {
        const data = this.getMetric(metric);
        if (data.length === 0) return 0;
        
        const sum = data.reduce((acc, point) => acc + point.value, 0);
        return sum / data.length;
    }

    calculateMax(metric) {
        const data = this.getMetric(metric);
        if (data.length === 0) return 0;
        
        return Math.max(...data.map(point => point.value));
    }

    calculateMin(metric) {
        const data = this.getMetric(metric);
        if (data.length === 0) return 0;
        
        return Math.min(...data.map(point => point.value));
    }
}

// ============================================================================
// Chart Manager
// ============================================================================

class ChartManager {
    constructor() {
        this.charts = {};
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        };
    }

    createLineChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const config = {
            type: 'line',
            data: data,
            options: { ...this.defaultOptions, ...options }
        };

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    }

    createBarChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const config = {
            type: 'bar',
            data: data,
            options: { ...this.defaultOptions, ...options }
        };

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    }

    createDoughnutChart(canvasId, data, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const config = {
            type: 'doughnut',
            data: data,
            options: { ...this.defaultOptions, ...options }
        };

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    }

    updateChart(canvasId, data) {
        const chart = this.charts[canvasId];
        if (!chart) {
            console.warn(`Chart ${canvasId} not found`);
            return;
        }

        chart.data = data;
        chart.update();
    }

    updateChartDataset(canvasId, datasetIndex, newData) {
        const chart = this.charts[canvasId];
        if (!chart) {
            console.warn(`Chart ${canvasId} not found`);
            return;
        }

        if (chart.data.datasets[datasetIndex]) {
            chart.data.datasets[datasetIndex].data = newData;
            chart.update();
        }
    }

    addDataPoint(canvasId, datasetIndex, label, value) {
        const chart = this.charts[canvasId];
        if (!chart) {
            console.warn(`Chart ${canvasId} not found`);
            return;
        }

        chart.data.labels.push(label);
        chart.data.datasets[datasetIndex].data.push(value);
        chart.update();
    }

    removeOldestDataPoint(canvasId, datasetIndex) {
        const chart = this.charts[canvasId];
        if (!chart) {
            console.warn(`Chart ${canvasId} not found`);
            return;
        }

        chart.data.labels.shift();
        chart.data.datasets[datasetIndex].data.shift();
        chart.update();
    }

    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }

    destroyAll() {
        Object.keys(this.charts).forEach(canvasId => {
            this.destroyChart(canvasId);
        });
    }
}

// ============================================================================
// Dashboard Controller
// ============================================================================

class DashboardController {
    constructor(config = {}) {
        this.config = {
            wsUrl: config.wsUrl || `ws://${window.location.host}/ws`,
            apiBaseUrl: config.apiBaseUrl || '/api',
            updateInterval: config.updateInterval || 5000,
            maxDataPoints: config.maxDataPoints || 50
        };

        this.api = new APIClient(this.config.apiBaseUrl);
        this.metricsManager = new MetricsManager(this.api);
        this.chartManager = new ChartManager();
        this.wsManager = null;
        this.updateTimer = null;
    }

    async initialize() {
        console.log('Initializing dashboard...');

        // Initialize WebSocket connection
        this.initializeWebSocket();

        // Load initial data
        await this.loadInitialData();

        // Setup charts
        this.setupCharts();

        // Start periodic updates
        this.startPeriodicUpdates();

        // Setup event listeners
        this.setupEventListeners();

        console.log('Dashboard initialized successfully');
    }

    initializeWebSocket() {
        this.wsManager = new WebSocketManager(this.config.wsUrl);

        this.wsManager.on('open', () => {
            this.updateConnectionStatus(true);
        });

        this.wsManager.on('message', (data) => {
            this.handleWebSocketMessage(data);
        });

        this.wsManager.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        });

        this.wsManager.on('close', () => {
            this.updateConnectionStatus(false);
        });

        this.wsManager.connect();
    }

    handleWebSocketMessage(data) {
        if (data.type === 'metrics') {
            this.updateMetrics(data.payload);
        } else if (data.type === 'alert') {
            this.handleAlert(data.payload);
        } else if (data.type === 'status') {
            this.updateSystemStatus(data.payload);
        }
    }

    async loadInitialData() {
        try {
            const [metrics, status, services] = await Promise.all([
                this.api.getMetrics(),
                this.api.getSystemStatus(),
                this.api.getServices()
            ]);

            this.metricsManager.processMetricsData(metrics);
            this.updateSystemStatus(status);
            this.updateServicesList(services);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    setupCharts() {
        // CPU Usage Chart
        this.chartManager.createLineChart('cpuChart', {
            labels: [],
            datasets: [{
                label: 'CPU Usage (%)',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.4
            }]
        }, {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        });

        // Memory Usage Chart
        this.chartManager.createLineChart('memoryChart', {
            labels: [],
            datasets: [{
                label: 'Memory Usage (%)',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.4
            }]
        }, {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        });

        // Network Traffic Chart
        this.chartManager.createLineChart('networkChart', {
            labels: [],
            datasets: [
                {
                    label: 'Inbound (MB/s)',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4
                },
                {
                    label: 'Outbound (MB/s)',
                    data: [],
                    borderColor: 'rgb(255, 206, 86)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.4
                }
            ]
        });
    }

    updateMetrics(metricsData) {
        const timestamp = new Date().toLocaleTimeString();

        // Update CPU
        if (metricsData.cpu !== undefined) {
            this.metricsManager.addDataPoint('cpu', timestamp, metricsData.cpu);
            this.updateCPUChart();
        }

        // Update Memory
        if (metricsData.memory !== undefined) {
            this.metricsManager.addDataPoint('memory', timestamp, metricsData.memory);
            this.updateMemoryChart();
        }

        // Update Network
        if (metricsData.network !== undefined) {
            this.metricsManager.addDataPoint('network', timestamp, metricsData.network);
            this.updateNetworkChart();
        }

        // Update metric displays
        this.updateMetricDisplays();
    }

    updateCPUChart() {
        const cpuData = this.metricsManager.getMetric('cpu');
        const labels = cpuData.map(point => point.timestamp);
        const values = cpuData.map(point => point.value);

        this.chartManager.updateChart('cpuChart', {
            labels: labels,
            datasets: [{
                label: 'CPU Usage (%)',
                data: values,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.4
            }]
        });
    }

    updateMemoryChart() {
        const memoryData = this.metricsManager.getMetric('memory');
        const labels = memoryData.map(point => point.timestamp);
        const values = memoryData.map(point => point.value);

        this.chartManager.updateChart('memoryChart', {
            labels: labels,
            datasets: [{
                label: 'Memory Usage (%)',
                data: values,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.4
            }]
        });
    }

    updateNetworkChart() {
        const networkData = this.metricsManager.getMetric('network');
        const labels = networkData.map(point => point.timestamp);
        const inbound = networkData.map(point => point.value.inbound || 0);
        const outbound = networkData.map(point => point.value.outbound || 0);

        this.chartManager.updateChart('networkChart', {
            labels: labels,
            datasets: [
                {
                    label: 'Inbound (MB/s)',
                    data: inbound,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4
                },
                {
                    label: 'Outbound (MB/s)',
                    data: outbound,
                    borderColor: 'rgb(255, 206, 86)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.4
                }
            ]
        });
    }

    updateMetricDisplays() {
        const cpuAvg = this.metricsManager.calculateAverage('cpu');
        const memoryAvg = this.metricsManager.calculateAverage('memory');

        this.updateElement('cpuValue', `${cpuAvg.toFixed(1)}%`);
        this.updateElement('memoryValue', `${memoryAvg.toFixed(1)}%`);
    }

    updateSystemStatus(status) {
        this.updateElement('systemStatus', status.overall || 'Unknown');
        this.updateElement('uptime', this.formatUptime(status.uptime || 0));
        this.updateElement('activeServices', status.activeServices || 0);
    }

    updateServicesList(services) {
        const container = document.getElementById('servicesList');
        if (!container) return;

        container.innerHTML = services.map(service => `
            <div class="service-item" data-service-id="${service.id}">
                <span class="service-name">${service.name}</span>
                <span class="service-status status-${service.status.toLowerCase()}">${service.status}</span>
            </div>
        `).join('');
    }

    handleAlert(alert) {
        console.warn('Alert received:', alert);
        this.showNotification(alert.message, alert.level || 'warning');
    }

    updateConnectionStatus(isConnected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = isConnected ? 'Connected' : 'Disconnected';
            statusElement.className = isConnected ? 'status-connected' : 'status-disconnected';
        }
    }

    startPeriodicUpdates() {
        this.updateTimer = setInterval(() => {
            this.refreshData();
        }, this.config.updateInterval);
    }

    stopPeriodicUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    async refreshData() {
        try {
            const [status, services] = await Promise.all([
                this.api.getSystemStatus(),
                this.api.getServices()
            ]);

            this.updateSystemStatus(status);
            this.updateServicesList(services);
        } catch (error) {
            console.error('Error refreshing data:', error);
        }
    }

    setupEventListeners() {
        // Time range selector
        const timeRangeSelect = document.getElementById('timeRange');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', (e) => {
                this.changeTimeRange(e.target.value);
            });
        }

        // Refresh button
        const refreshButton = document.getElementById('refreshBtn');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                this.refreshData();
            });
        }
    }

    async changeTimeRange(range) {
        try {
            const metrics = await this.api.getMetrics(range);
            this.metricsManager.clearAll();
            this.metricsManager.processMetricsData(metrics);
            this.updateCPUChart();
            this.updateMemoryChart();
            this.updateNetworkChart();
        } catch (error) {
            console.error('Error changing time range:', error);
        }
    }

    // Utility methods
    updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = content;
        }
    }

    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${days}d ${hours}h ${minutes}m`;
    }

    showNotification(message, level = 'info') {
        // Implement notification display logic
        console.log(`[${level.toUpperCase()}] ${message}`);
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    destroy() {
        this.stopPeriodicUpdates();
        if (this.wsManager) {
            this.wsManager.close();
        }
        this.chartManager.destroyAll();
    }
}

// ============================================================================
// Initialize Dashboard on Page Load
// ============================================================================

let dashboard = null;

document.addEventListener('DOMContentLoaded', () => {
    const config = {
        wsUrl: window.DASHBOARD_CONFIG?.wsUrl || `ws://${window.location.host}/ws`,
        apiBaseUrl: window.DASHBOARD_CONFIG?.apiBaseUrl || '/api',
        updateInterval: window.DASHBOARD_CONFIG?.updateInterval || 5000,
        maxDataPoints: window.DASHBOARD_CONFIG?.maxDataPoints || 50
    };

    dashboard = new DashboardController(config);
    dashboard.initialize().catch(error => {
        console.error('Failed to initialize dashboard:', error);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboard) {
        dashboard.destroy();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WebSocketManager,
        APIClient,
        MetricsManager,
        ChartManager,
        DashboardController
    };
}
