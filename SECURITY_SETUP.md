# Security Setup Guide

**Last Updated:** 2025-12-11  
**Repository:** jetgause/Library-

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Security Configuration](#security-configuration)
3. [Common Errors and Solutions](#common-errors-and-solutions)
4. [Deployment Configurations](#deployment-configurations)
5. [Verification Checklist](#verification-checklist)
6. [Security Best Practices](#security-best-practices)
7. [Additional Resources](#additional-resources)

---

## Quick Start Guide

### Prerequisites

Before implementing security measures, ensure you have:

- Administrative access to the repository
- Understanding of your application's architecture
- Access to environment variable configuration
- Familiarity with your deployment platform

### Initial Setup (5 Minutes)

1. **Enable Security Features**
   ```bash
   # Clone the repository
   git clone https://github.com/jetgause/Library-.git
   cd Library-
   
   # Install security dependencies
   npm install --save helmet express-rate-limit bcrypt jsonwebtoken
   # or
   pip install cryptography pyjwt python-dotenv
   ```

2. **Configure Environment Variables**
   ```bash
   # Create .env file (never commit this!)
   cp .env.example .env
   
   # Edit with your secure values
   nano .env
   ```

3. **Enable GitHub Security Features**
   - Navigate to Settings → Security & analysis
   - Enable Dependabot alerts
   - Enable Dependabot security updates
   - Enable Secret scanning
   - Enable Code scanning (GitHub Advanced Security)

---

## Security Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Application Security
NODE_ENV=production
SECRET_KEY=your-256-bit-secret-key-here
JWT_SECRET=your-jwt-secret-key-here
SESSION_SECRET=your-session-secret-here

# Database Security
DB_HOST=localhost
DB_PORT=5432
DB_NAME=library_db
DB_USER=db_user
DB_PASSWORD=strong-database-password
DB_SSL=true

# API Keys (never expose in client-side code)
API_KEY=your-api-key-here
ENCRYPTION_KEY=your-encryption-key-here

# CORS Configuration
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# Security Headers
HSTS_MAX_AGE=31536000
CSP_DIRECTIVES=default-src 'self'; script-src 'self' 'unsafe-inline'
```

### Security Middleware Configuration

#### Node.js/Express Example

```javascript
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const express = require('express');

const app = express();

// Security Headers
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

// Rate Limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});

app.use('/api/', limiter);

// CORS Configuration
const cors = require('cors');
const allowedOrigins = process.env.ALLOWED_ORIGINS.split(',');

app.use(cors({
  origin: function(origin, callback) {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));
```

#### Python/Flask Example

```python
from flask import Flask
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

app = Flask(__name__)

# Security Headers
csp = {
    'default-src': "'self'",
    'script-src': "'self'",
    'style-src': "'self' 'unsafe-inline'"
}

Talisman(app, 
         content_security_policy=csp,
         force_https=True,
         strict_transport_security=True,
         strict_transport_security_max_age=31536000)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

# Secret Key Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
```

### Authentication Configuration

```javascript
// JWT Token Configuration
const jwt = require('jsonwebtoken');

const generateToken = (userId) => {
  return jwt.sign(
    { userId: userId },
    process.env.JWT_SECRET,
    { 
      expiresIn: '24h',
      issuer: 'library-app',
      audience: 'library-users'
    }
  );
};

const verifyToken = (token) => {
  try {
    return jwt.verify(token, process.env.JWT_SECRET);
  } catch (error) {
    throw new Error('Invalid token');
  }
};
```

### Password Security

```javascript
const bcrypt = require('bcrypt');

// Password Hashing
async function hashPassword(password) {
  const saltRounds = 12;
  return await bcrypt.hash(password, saltRounds);
}

// Password Verification
async function verifyPassword(password, hash) {
  return await bcrypt.compare(password, hash);
}

// Password Requirements
const passwordRequirements = {
  minLength: 12,
  requireUppercase: true,
  requireLowercase: true,
  requireNumbers: true,
  requireSpecialChars: true
};
```

---

## Common Errors and Solutions

### Error 1: "Environment variable not found"

**Symptom:** Application crashes with `undefined` environment variables

**Solution:**
```bash
# Ensure .env file exists and is properly formatted
cp .env.example .env

# Verify variables are loaded
node -e "require('dotenv').config(); console.log(process.env.SECRET_KEY)"

# For production, set environment variables in your hosting platform
```

### Error 2: "CORS policy blocking requests"

**Symptom:** Browser console shows CORS errors

**Solution:**
```javascript
// Update CORS configuration to include your domain
const corsOptions = {
  origin: ['https://yourdomain.com', 'http://localhost:3000'],
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
```

### Error 3: "Rate limit exceeded"

**Symptom:** Users receive 429 Too Many Requests errors

**Solution:**
```javascript
// Adjust rate limit based on your needs
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 200, // Increase if legitimate users hit the limit
  skip: (req) => {
    // Skip rate limiting for trusted IPs
    return req.ip === 'trusted.ip.address';
  }
});
```

### Error 4: "JWT token expired"

**Symptom:** Users logged out unexpectedly

**Solution:**
```javascript
// Implement token refresh mechanism
app.post('/api/refresh-token', async (req, res) => {
  const refreshToken = req.cookies.refreshToken;
  
  try {
    const decoded = jwt.verify(refreshToken, process.env.REFRESH_TOKEN_SECRET);
    const newAccessToken = generateToken(decoded.userId);
    
    res.json({ accessToken: newAccessToken });
  } catch (error) {
    res.status(401).json({ error: 'Invalid refresh token' });
  }
});
```

### Error 5: "Database connection SSL error"

**Symptom:** Cannot connect to database in production

**Solution:**
```javascript
// Configure SSL for database connection
const dbConfig = {
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  ssl: {
    rejectUnauthorized: true,
    ca: fs.readFileSync('./ca-certificate.crt').toString()
  }
};
```

---

## Deployment Configurations

### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - NODE_ENV=development
      - DEBUG=true
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
```

### Staging Environment

```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - NODE_ENV=staging
      - RATE_LIMIT_MAX_REQUESTS=500
    env_file:
      - .env.staging
    ports:
      - "8080:8080"
```

### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - NODE_ENV=production
    env_file:
      - .env.production
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Heroku Deployment

```bash
# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set JWT_SECRET=your-jwt-secret
heroku config:set NODE_ENV=production

# Enable SSL
heroku labs:enable http-sni
```

### AWS Deployment

```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name library-app-secrets \
  --secret-string file://secrets.json

# IAM policy for accessing secrets
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "secretsmanager:GetSecretValue",
    "Resource": "arn:aws:secretsmanager:region:account-id:secret:library-app-secrets*"
  }]
}
```

### Kubernetes Deployment

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: library-app-secrets
type: Opaque
data:
  secret-key: base64-encoded-secret
  jwt-secret: base64-encoded-jwt-secret

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: library-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: library-app:latest
        envFrom:
        - secretRef:
            name: library-app-secrets
        ports:
        - containerPort: 8080
```

---

## Verification Checklist

### Pre-Deployment Checklist

- [ ] All sensitive data stored in environment variables
- [ ] `.env` file added to `.gitignore`
- [ ] No hardcoded secrets in source code
- [ ] Security dependencies up to date
- [ ] Rate limiting configured and tested
- [ ] CORS properly configured for allowed origins
- [ ] HTTPS/SSL certificates configured
- [ ] Security headers enabled (Helmet.js or equivalent)
- [ ] Input validation implemented
- [ ] SQL injection prevention measures in place
- [ ] XSS protection enabled
- [ ] CSRF tokens implemented for state-changing operations
- [ ] Authentication middleware protecting sensitive routes
- [ ] Password hashing with bcrypt (12+ rounds)
- [ ] JWT tokens with expiration times
- [ ] Database connections using SSL
- [ ] Error messages don't leak sensitive information
- [ ] Logging configured (exclude sensitive data)
- [ ] Security scanning tools run (npm audit, Snyk, etc.)

### Post-Deployment Checklist

- [ ] SSL/TLS certificate valid and properly configured
- [ ] Security headers present (check with securityheaders.com)
- [ ] Rate limiting working as expected
- [ ] CORS policy prevents unauthorized origins
- [ ] Authentication flow working correctly
- [ ] Session management secure
- [ ] API endpoints properly protected
- [ ] Database connections secure
- [ ] Logs being collected and monitored
- [ ] Automated security updates enabled
- [ ] Backup and recovery procedures tested
- [ ] Incident response plan documented
- [ ] Security monitoring alerts configured

### Testing Checklist

- [ ] Penetration testing completed
- [ ] SQL injection tests passed
- [ ] XSS vulnerability tests passed
- [ ] CSRF protection tests passed
- [ ] Authentication bypass tests passed
- [ ] Authorization tests passed
- [ ] Session management tests passed
- [ ] Rate limiting tests passed
- [ ] Input validation tests passed
- [ ] Error handling tests passed

---

## Security Best Practices

### 1. Authentication & Authorization

**Do:**
- ✅ Use industry-standard authentication (OAuth 2.0, JWT)
- ✅ Implement multi-factor authentication (MFA)
- ✅ Use secure session management
- ✅ Implement proper password policies
- ✅ Use role-based access control (RBAC)

**Don't:**
- ❌ Store passwords in plain text
- ❌ Use weak hashing algorithms (MD5, SHA1)
- ❌ Implement custom authentication without expertise
- ❌ Trust client-side authorization checks alone

### 2. Data Protection

**Encryption at Rest:**
```javascript
const crypto = require('crypto');

function encrypt(text) {
  const algorithm = 'aes-256-gcm';
  const key = Buffer.from(process.env.ENCRYPTION_KEY, 'hex');
  const iv = crypto.randomBytes(16);
  
  const cipher = crypto.createCipheriv(algorithm, key, iv);
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  
  const authTag = cipher.getAuthTag();
  
  return {
    encrypted: encrypted,
    iv: iv.toString('hex'),
    authTag: authTag.toString('hex')
  };
}
```

**Encryption in Transit:**
- Always use HTTPS/TLS 1.2+
- Configure strong cipher suites
- Use HSTS headers
- Implement certificate pinning for mobile apps

### 3. Input Validation

```javascript
const validator = require('validator');

function validateUserInput(input) {
  // Sanitize input
  const sanitized = validator.escape(input);
  
  // Validate format
  if (!validator.isLength(sanitized, { min: 1, max: 255 })) {
    throw new Error('Invalid input length');
  }
  
  // Check for SQL injection patterns
  if (/(\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)/i.test(sanitized)) {
    throw new Error('Potentially malicious input detected');
  }
  
  return sanitized;
}
```

### 4. Dependency Management

```bash
# Regular security audits
npm audit
npm audit fix

# Use Snyk for continuous monitoring
npm install -g snyk
snyk test
snyk monitor

# Keep dependencies updated
npm outdated
npm update
```

### 5. Logging & Monitoring

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Security event logging
function logSecurityEvent(event, details) {
  logger.warn('Security Event', {
    timestamp: new Date().toISOString(),
    event: event,
    details: details,
    // Never log: passwords, tokens, credit cards, PII
  });
}
```

### 6. Error Handling

```javascript
// Production error handler
app.use((err, req, res, next) => {
  // Log detailed error server-side
  logger.error('Error:', {
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method
  });
  
  // Send generic error to client
  res.status(err.status || 500).json({
    error: 'An error occurred',
    // Don't expose: stack traces, internal paths, database errors
  });
});
```

### 7. API Security

```javascript
// API Key middleware
function validateApiKey(req, res, next) {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey || apiKey !== process.env.API_KEY) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  next();
}

// Request signing for sensitive operations
function verifyRequestSignature(req, res, next) {
  const signature = req.headers['x-signature'];
  const timestamp = req.headers['x-timestamp'];
  
  // Verify timestamp to prevent replay attacks
  const requestAge = Date.now() - parseInt(timestamp);
  if (requestAge > 300000) { // 5 minutes
    return res.status(401).json({ error: 'Request expired' });
  }
  
  // Verify signature
  const expectedSignature = crypto
    .createHmac('sha256', process.env.SECRET_KEY)
    .update(timestamp + JSON.stringify(req.body))
    .digest('hex');
  
  if (signature !== expectedSignature) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  next();
}
```

### 8. Database Security

```javascript
// Use parameterized queries
const query = 'SELECT * FROM users WHERE email = $1';
const values = [userEmail];
db.query(query, values);

// Don't concatenate user input
// ❌ BAD: const query = `SELECT * FROM users WHERE email = '${userEmail}'`;

// Use ORM with built-in protections
const User = require('./models/User');
const user = await User.findOne({ where: { email: userEmail } });
```

### 9. Regular Security Updates

```bash
# Set up automated security updates
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "jetgause"
    labels:
      - "dependencies"
      - "security"
```

### 10. Incident Response Plan

1. **Detection:** Monitor logs and alerts
2. **Assessment:** Determine severity and impact
3. **Containment:** Isolate affected systems
4. **Eradication:** Remove threat and vulnerabilities
5. **Recovery:** Restore normal operations
6. **Lessons Learned:** Document and improve

---

## Additional Resources

### Security Tools

- **OWASP ZAP:** Web application security scanner
- **Burp Suite:** Security testing platform
- **Snyk:** Dependency vulnerability scanning
- **SonarQube:** Code quality and security analysis
- **GitHub Advanced Security:** Secret scanning and code scanning

### Documentation

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Node.js Security Best Practices](https://nodejs.org/en/docs/guides/security/)
- [Express Security Best Practices](https://expressjs.com/en/advanced/best-practice-security.html)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)

### Security Headers

Test your security headers at:
- https://securityheaders.com
- https://observatory.mozilla.org

### Compliance

- **GDPR:** EU data protection regulation
- **CCPA:** California Consumer Privacy Act
- **HIPAA:** Health Insurance Portability and Accountability Act
- **PCI DSS:** Payment Card Industry Data Security Standard

---

## Support & Contact

For security concerns or to report vulnerabilities:

- **Security Email:** security@yourdomain.com
- **Bug Bounty:** (if applicable)
- **Response Time:** Within 24 hours for critical issues

**Remember:** Never commit sensitive information to version control!

---

*This document should be reviewed and updated quarterly or when significant security changes occur.*