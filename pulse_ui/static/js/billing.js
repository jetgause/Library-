/**
 * Billing Integration Module
 * Handles Google Pay and In-App Billing with purchase verification,
 * subscription management, and receipt validation
 * 
 * @author jetgause
 * @version 1.0.0
 * @date 2025-12-10
 */

class BillingManager {
    constructor(config = {}) {
        this.config = {
            environment: config.environment || 'TEST',
            merchantId: config.merchantId || '',
            merchantName: config.merchantName || 'Library App',
            apiEndpoint: config.apiEndpoint || '/api/billing',
            googlePayVersion: 2,
            googlePayVersionMinor: 0,
            ...config
        };

        this.googlePayClient = null;
        this.purchaseHistory = [];
        this.subscriptions = [];
        this.isGooglePayReady = false;
        
        this.init();
    }

    /**
     * Initialize billing system
     */
    async init() {
        try {
            await this.initGooglePay();
            await this.loadPurchaseHistory();
            await this.loadSubscriptions();
            this.setupEventListeners();
            console.log('Billing Manager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize billing manager:', error);
            throw error;
        }
    }

    /**
     * Initialize Google Pay
     */
    async initGooglePay() {
        if (!window.google || !window.google.payments) {
            console.warn('Google Pay SDK not loaded');
            return;
        }

        const baseRequest = {
            apiVersion: this.config.googlePayVersion,
            apiVersionMinor: this.config.googlePayVersionMinor
        };

        this.googlePayClient = new google.payments.api.PaymentsClient({
            environment: this.config.environment
        });

        try {
            const isReadyToPay = await this.googlePayClient.isReadyToPay({
                ...baseRequest,
                allowedPaymentMethods: this.getAllowedPaymentMethods()
            });

            this.isGooglePayReady = isReadyToPay.result;
            
            if (this.isGooglePayReady) {
                this.renderGooglePayButton();
            }
        } catch (error) {
            console.error('Error checking Google Pay availability:', error);
        }
    }

    /**
     * Get allowed payment methods for Google Pay
     */
    getAllowedPaymentMethods() {
        return [
            {
                type: 'CARD',
                parameters: {
                    allowedAuthMethods: ['PAN_ONLY', 'CRYPTOGRAM_3DS'],
                    allowedCardNetworks: ['AMEX', 'DISCOVER', 'MASTERCARD', 'VISA']
                },
                tokenizationSpecification: {
                    type: 'PAYMENT_GATEWAY',
                    parameters: {
                        gateway: 'example',
                        gatewayMerchantId: this.config.merchantId
                    }
                }
            }
        ];
    }

    /**
     * Get Google Pay payment data request
     */
    getGooglePaymentDataRequest(transactionInfo) {
        return {
            apiVersion: this.config.googlePayVersion,
            apiVersionMinor: this.config.googlePayVersionMinor,
            allowedPaymentMethods: this.getAllowedPaymentMethods(),
            merchantInfo: {
                merchantId: this.config.merchantId,
                merchantName: this.config.merchantName
            },
            transactionInfo: transactionInfo
        };
    }

    /**
     * Render Google Pay button
     */
    renderGooglePayButton() {
        const buttonContainer = document.getElementById('google-pay-button');
        if (!buttonContainer || !this.googlePayClient) return;

        const button = this.googlePayClient.createButton({
            onClick: () => this.onGooglePaymentButtonClicked(),
            buttonColor: 'default',
            buttonType: 'buy',
            buttonSizeMode: 'fill'
        });

        buttonContainer.innerHTML = '';
        buttonContainer.appendChild(button);
    }

    /**
     * Handle Google Pay button click
     */
    async onGooglePaymentButtonClicked() {
        const transactionInfo = this.getTransactionInfo();
        const paymentDataRequest = this.getGooglePaymentDataRequest(transactionInfo);

        try {
            const paymentData = await this.googlePayClient.loadPaymentData(paymentDataRequest);
            await this.processGooglePayPayment(paymentData);
        } catch (error) {
            console.error('Google Pay payment error:', error);
            this.handlePaymentError(error);
        }
    }

    /**
     * Get transaction info
     */
    getTransactionInfo() {
        return {
            displayItems: [
                {
                    label: 'Subtotal',
                    type: 'SUBTOTAL',
                    price: '10.00'
                },
                {
                    label: 'Tax',
                    type: 'TAX',
                    price: '1.00'
                }
            ],
            countryCode: 'US',
            currencyCode: 'USD',
            totalPriceStatus: 'FINAL',
            totalPrice: '11.00',
            totalPriceLabel: 'Total'
        };
    }

    /**
     * Process Google Pay payment
     */
    async processGooglePayPayment(paymentData) {
        try {
            const token = paymentData.paymentMethodData.tokenizationData.token;
            
            const response = await fetch(`${this.config.apiEndpoint}/process-payment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    paymentToken: token,
                    paymentData: paymentData,
                    timestamp: new Date().toISOString()
                })
            });

            const result = await response.json();

            if (result.success) {
                await this.verifyPurchase(result.purchaseToken);
                this.handlePaymentSuccess(result);
            } else {
                throw new Error(result.message || 'Payment processing failed');
            }
        } catch (error) {
            console.error('Payment processing error:', error);
            throw error;
        }
    }

    /**
     * Verify purchase with server
     */
    async verifyPurchase(purchaseToken) {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/verify-purchase`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    purchaseToken: purchaseToken,
                    timestamp: new Date().toISOString()
                })
            });

            const verification = await response.json();

            if (!verification.valid) {
                throw new Error('Purchase verification failed');
            }

            return verification;
        } catch (error) {
            console.error('Purchase verification error:', error);
            throw error;
        }
    }

    /**
     * Validate receipt
     */
    async validateReceipt(receipt) {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/validate-receipt`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    receipt: receipt,
                    platform: this.detectPlatform(),
                    timestamp: new Date().toISOString()
                })
            });

            const validation = await response.json();

            if (validation.valid) {
                await this.storePurchaseLocally(validation.purchase);
            }

            return validation;
        } catch (error) {
            console.error('Receipt validation error:', error);
            throw error;
        }
    }

    /**
     * Purchase product
     */
    async purchaseProduct(productId, productType = 'inapp') {
        try {
            // Request purchase
            const purchase = await this.requestPurchase(productId, productType);
            
            // Verify purchase
            const verification = await this.verifyPurchase(purchase.purchaseToken);
            
            // Validate receipt
            const validation = await this.validateReceipt(purchase.receipt);
            
            if (validation.valid) {
                // Acknowledge purchase
                await this.acknowledgePurchase(purchase.purchaseToken);
                
                // Update local records
                await this.storePurchaseLocally(purchase);
                
                this.handlePurchaseSuccess(purchase);
                return purchase;
            } else {
                throw new Error('Purchase validation failed');
            }
        } catch (error) {
            console.error('Purchase error:', error);
            this.handlePurchaseError(error);
            throw error;
        }
    }

    /**
     * Request purchase from platform
     */
    async requestPurchase(productId, productType) {
        const response = await fetch(`${this.config.apiEndpoint}/request-purchase`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                productId: productId,
                productType: productType,
                timestamp: new Date().toISOString()
            })
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.message || 'Purchase request failed');
        }

        return result.purchase;
    }

    /**
     * Acknowledge purchase
     */
    async acknowledgePurchase(purchaseToken) {
        const response = await fetch(`${this.config.apiEndpoint}/acknowledge-purchase`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                purchaseToken: purchaseToken,
                timestamp: new Date().toISOString()
            })
        });

        return await response.json();
    }

    /**
     * Subscribe to product
     */
    async subscribe(subscriptionId, offerToken = null) {
        try {
            const subscription = await this.requestSubscription(subscriptionId, offerToken);
            
            const verification = await this.verifyPurchase(subscription.purchaseToken);
            
            if (verification.valid) {
                await this.storeSubscriptionLocally(subscription);
                this.handleSubscriptionSuccess(subscription);
                return subscription;
            } else {
                throw new Error('Subscription verification failed');
            }
        } catch (error) {
            console.error('Subscription error:', error);
            this.handleSubscriptionError(error);
            throw error;
        }
    }

    /**
     * Request subscription from platform
     */
    async requestSubscription(subscriptionId, offerToken) {
        const response = await fetch(`${this.config.apiEndpoint}/request-subscription`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                subscriptionId: subscriptionId,
                offerToken: offerToken,
                timestamp: new Date().toISOString()
            })
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.message || 'Subscription request failed');
        }

        return result.subscription;
    }

    /**
     * Cancel subscription
     */
    async cancelSubscription(subscriptionId) {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/cancel-subscription`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    subscriptionId: subscriptionId,
                    timestamp: new Date().toISOString()
                })
            });

            const result = await response.json();

            if (result.success) {
                await this.removeSubscriptionLocally(subscriptionId);
                this.handleCancellationSuccess(subscriptionId);
            }

            return result;
        } catch (error) {
            console.error('Subscription cancellation error:', error);
            throw error;
        }
    }

    /**
     * Update subscription
     */
    async updateSubscription(oldSubscriptionId, newSubscriptionId, prorationMode = 'IMMEDIATE_WITH_TIME_PRORATION') {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/update-subscription`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    oldSubscriptionId: oldSubscriptionId,
                    newSubscriptionId: newSubscriptionId,
                    prorationMode: prorationMode,
                    timestamp: new Date().toISOString()
                })
            });

            const result = await response.json();

            if (result.success) {
                await this.updateSubscriptionLocally(result.subscription);
                this.handleSubscriptionUpdateSuccess(result.subscription);
            }

            return result;
        } catch (error) {
            console.error('Subscription update error:', error);
            throw error;
        }
    }

    /**
     * Get subscription status
     */
    async getSubscriptionStatus(subscriptionId) {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/subscription-status/${subscriptionId}`, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            return await response.json();
        } catch (error) {
            console.error('Error fetching subscription status:', error);
            throw error;
        }
    }

    /**
     * Restore purchases
     */
    async restorePurchases() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/restore-purchases`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({
                    timestamp: new Date().toISOString()
                })
            });

            const result = await response.json();

            if (result.success) {
                this.purchaseHistory = result.purchases || [];
                this.subscriptions = result.subscriptions || [];
                
                await this.syncPurchasesLocally();
                this.handleRestoreSuccess(result);
            }

            return result;
        } catch (error) {
            console.error('Restore purchases error:', error);
            throw error;
        }
    }

    /**
     * Load purchase history
     */
    async loadPurchaseHistory() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/purchase-history`, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const result = await response.json();
            this.purchaseHistory = result.purchases || [];
            return this.purchaseHistory;
        } catch (error) {
            console.error('Error loading purchase history:', error);
            return [];
        }
    }

    /**
     * Load subscriptions
     */
    async loadSubscriptions() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}/subscriptions`, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            const result = await response.json();
            this.subscriptions = result.subscriptions || [];
            return this.subscriptions;
        } catch (error) {
            console.error('Error loading subscriptions:', error);
            return [];
        }
    }

    /**
     * Store purchase locally
     */
    async storePurchaseLocally(purchase) {
        this.purchaseHistory.push(purchase);
        
        try {
            localStorage.setItem('billing_purchases', JSON.stringify(this.purchaseHistory));
        } catch (error) {
            console.error('Error storing purchase locally:', error);
        }
    }

    /**
     * Store subscription locally
     */
    async storeSubscriptionLocally(subscription) {
        const existingIndex = this.subscriptions.findIndex(s => s.id === subscription.id);
        
        if (existingIndex >= 0) {
            this.subscriptions[existingIndex] = subscription;
        } else {
            this.subscriptions.push(subscription);
        }
        
        try {
            localStorage.setItem('billing_subscriptions', JSON.stringify(this.subscriptions));
        } catch (error) {
            console.error('Error storing subscription locally:', error);
        }
    }

    /**
     * Update subscription locally
     */
    async updateSubscriptionLocally(subscription) {
        await this.storeSubscriptionLocally(subscription);
    }

    /**
     * Remove subscription locally
     */
    async removeSubscriptionLocally(subscriptionId) {
        this.subscriptions = this.subscriptions.filter(s => s.id !== subscriptionId);
        
        try {
            localStorage.setItem('billing_subscriptions', JSON.stringify(this.subscriptions));
        } catch (error) {
            console.error('Error removing subscription locally:', error);
        }
    }

    /**
     * Sync purchases locally
     */
    async syncPurchasesLocally() {
        try {
            localStorage.setItem('billing_purchases', JSON.stringify(this.purchaseHistory));
            localStorage.setItem('billing_subscriptions', JSON.stringify(this.subscriptions));
        } catch (error) {
            console.error('Error syncing purchases locally:', error);
        }
    }

    /**
     * Check if user owns product
     */
    ownsProduct(productId) {
        return this.purchaseHistory.some(p => p.productId === productId && p.status === 'active');
    }

    /**
     * Check if user has active subscription
     */
    hasActiveSubscription(subscriptionId = null) {
        if (subscriptionId) {
            const subscription = this.subscriptions.find(s => s.id === subscriptionId);
            return subscription && subscription.status === 'active';
        }
        
        return this.subscriptions.some(s => s.status === 'active');
    }

    /**
     * Get active subscriptions
     */
    getActiveSubscriptions() {
        return this.subscriptions.filter(s => s.status === 'active');
    }

    /**
     * Detect platform
     */
    detectPlatform() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        
        if (/android/i.test(userAgent)) {
            return 'android';
        }
        
        if (/iPad|iPhone|iPod/.test(userAgent) && !window.MSStream) {
            return 'ios';
        }
        
        return 'web';
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Listen for purchase buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('purchase-button')) {
                const productId = e.target.dataset.productId;
                const productType = e.target.dataset.productType || 'inapp';
                this.purchaseProduct(productId, productType);
            }
            
            if (e.target.classList.contains('subscribe-button')) {
                const subscriptionId = e.target.dataset.subscriptionId;
                const offerToken = e.target.dataset.offerToken || null;
                this.subscribe(subscriptionId, offerToken);
            }
        });
    }

    /**
     * Handle payment success
     */
    handlePaymentSuccess(result) {
        this.dispatchEvent('paymentSuccess', result);
        console.log('Payment successful:', result);
    }

    /**
     * Handle payment error
     */
    handlePaymentError(error) {
        this.dispatchEvent('paymentError', error);
        console.error('Payment error:', error);
    }

    /**
     * Handle purchase success
     */
    handlePurchaseSuccess(purchase) {
        this.dispatchEvent('purchaseSuccess', purchase);
        console.log('Purchase successful:', purchase);
    }

    /**
     * Handle purchase error
     */
    handlePurchaseError(error) {
        this.dispatchEvent('purchaseError', error);
        console.error('Purchase error:', error);
    }

    /**
     * Handle subscription success
     */
    handleSubscriptionSuccess(subscription) {
        this.dispatchEvent('subscriptionSuccess', subscription);
        console.log('Subscription successful:', subscription);
    }

    /**
     * Handle subscription error
     */
    handleSubscriptionError(error) {
        this.dispatchEvent('subscriptionError', error);
        console.error('Subscription error:', error);
    }

    /**
     * Handle subscription update success
     */
    handleSubscriptionUpdateSuccess(subscription) {
        this.dispatchEvent('subscriptionUpdateSuccess', subscription);
        console.log('Subscription updated successfully:', subscription);
    }

    /**
     * Handle cancellation success
     */
    handleCancellationSuccess(subscriptionId) {
        this.dispatchEvent('cancellationSuccess', { subscriptionId });
        console.log('Subscription cancelled successfully:', subscriptionId);
    }

    /**
     * Handle restore success
     */
    handleRestoreSuccess(result) {
        this.dispatchEvent('restoreSuccess', result);
        console.log('Purchases restored successfully:', result);
    }

    /**
     * Dispatch custom event
     */
    dispatchEvent(eventName, detail) {
        const event = new CustomEvent(`billing:${eventName}`, { detail });
        document.dispatchEvent(event);
    }

    /**
     * Destroy billing manager
     */
    destroy() {
        this.googlePayClient = null;
        this.purchaseHistory = [];
        this.subscriptions = [];
        this.isGooglePayReady = false;
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BillingManager;
}

// Make available globally
window.BillingManager = BillingManager;
