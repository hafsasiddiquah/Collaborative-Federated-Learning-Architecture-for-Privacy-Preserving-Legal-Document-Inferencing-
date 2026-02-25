#!/bin/bash
#
# AWS EC2 Setup Script for Secure Federated Learning Aggregation Server
# This script configures an EC2 instance to run the secure aggregation server
# Production-grade setup for research deployment
#

set -e  # Exit on error

echo "=========================================="
echo "Secure FL Aggregation Server Setup"
echo "=========================================="

# Configuration variables
PYTHON_VERSION="3.9"
PROJECT_DIR="/opt/secure_legal_fl"
SERVICE_USER="flserver"
LOG_DIR="/var/log/secure_fl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_status "Starting EC2 setup..."

# Update system packages
print_status "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    openssl \
    nginx \
    ufw \
    supervisor \
    htop \
    vim \
    net-tools

# Install CUDA toolkit (if GPU instance)
if lspci | grep -i nvidia > /dev/null; then
    print_status "NVIDIA GPU detected. Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -y
    apt-get install -y cuda-toolkit-12-2
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    print_status "CUDA toolkit installed"
else
    print_warning "No NVIDIA GPU detected. Skipping CUDA installation."
fi

# Create service user
print_status "Creating service user: $SERVICE_USER"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d "$PROJECT_DIR" -m "$SERVICE_USER"
    print_status "User $SERVICE_USER created"
else
    print_warning "User $SERVICE_USER already exists"
fi

# Create project directory
print_status "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."
sudo -u "$SERVICE_USER" python3 -m venv "$PROJECT_DIR/venv"
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install --upgrade pip setuptools wheel

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
else
    print_warning "requirements.txt not found. Installing core dependencies..."
    sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install \
        torch transformers datasets accelerate \
        flwr peft bitsandbytes opacus \
        cryptography pyopenssl \
        rouge-score sacrebleu nltk \
        numpy pandas scipy tqdm pyyaml \
        boto3
fi

# Download NLTK data
print_status "Downloading NLTK data..."
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/python" -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded')
"

# Generate TLS certificates (self-signed for testing)
print_status "Generating TLS certificates..."
CERT_DIR="$PROJECT_DIR/certs"
mkdir -p "$CERT_DIR"

# Generate CA key and certificate
openssl genrsa -out "$CERT_DIR/ca-key.pem" 4096
openssl req -new -x509 -days 365 -key "$CERT_DIR/ca-key.pem" \
    -out "$CERT_DIR/ca-cert.pem" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=FL-CA"

# Generate server key
openssl genrsa -out "$CERT_DIR/server-key.pem" 4096

# Generate server certificate signing request
openssl req -new -key "$CERT_DIR/server-key.pem" \
    -out "$CERT_DIR/server.csr" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=fl-server"

# Generate server certificate
openssl x509 -req -days 365 -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca-cert.pem" \
    -CAkey "$CERT_DIR/ca-key.pem" \
    -CAcreateserial \
    -out "$CERT_DIR/server-cert.pem"

chmod 600 "$CERT_DIR"/*.pem
chown -R "$SERVICE_USER:$SERVICE_USER" "$CERT_DIR"

print_status "TLS certificates generated in $CERT_DIR"

# Configure firewall
print_status "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp  # SSH
ufw allow 8080/tcp  # FL Server
ufw allow 80/tcp  # HTTP (for health checks)
ufw allow 443/tcp  # HTTPS
print_status "Firewall configured"

# Create systemd service file
print_status "Creating systemd service..."
cat > /etc/systemd/system/secure-fl-server.service << EOF
[Unit]
Description=Secure Federated Learning Aggregation Server
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python -m server.secure_agg_server \\
    --address 0.0.0.0:8080 \\
    --rounds 10 \\
    --min-clients 2 \\
    --cert $CERT_DIR/server-cert.pem \\
    --key $CERT_DIR/server-key.pem \\
    --similarity-threshold 0.5
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/server.log
StandardError=append:$LOG_DIR/server.error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
print_status "Systemd service created"

# Create supervisor config (alternative to systemd)
print_status "Creating supervisor configuration..."
cat > /etc/supervisor/conf.d/secure-fl-server.conf << EOF
[program:secure-fl-server]
command=$PROJECT_DIR/venv/bin/python -m server.secure_agg_server --address 0.0.0.0:8080 --rounds 10 --min-clients 2 --cert $CERT_DIR/server-cert.pem --key $CERT_DIR/server-key.pem
directory=$PROJECT_DIR
user=$SERVICE_USER
autostart=true
autorestart=true
stderr_logfile=$LOG_DIR/server.error.log
stdout_logfile=$LOG_DIR/server.log
environment=PATH="$PROJECT_DIR/venv/bin"
EOF

supervisorctl reread
supervisorctl update
print_status "Supervisor configuration created"

# Create health check script
print_status "Creating health check script..."
cat > "$PROJECT_DIR/health_check.sh" << 'EOF'
#!/bin/bash
# Health check script for FL server
PORT=8080
if nc -z localhost $PORT; then
    echo "OK: Server is running on port $PORT"
    exit 0
else
    echo "FAIL: Server is not responding on port $PORT"
    exit 1
fi
EOF

chmod +x "$PROJECT_DIR/health_check.sh"
chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/health_check.sh"

# Create startup script
print_status "Creating startup script..."
cat > "$PROJECT_DIR/start_server.sh" << EOF
#!/bin/bash
cd $PROJECT_DIR
source venv/bin/activate
python -m server.secure_agg_server \\
    --address 0.0.0.0:8080 \\
    --rounds 10 \\
    --min-clients 2 \\
    --cert certs/server-cert.pem \\
    --key certs/server-key.pem \\
    --similarity-threshold 0.5
EOF

chmod +x "$PROJECT_DIR/start_server.sh"
chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/start_server.sh"

# Setup log rotation
print_status "Setting up log rotation..."
cat > /etc/logrotate.d/secure-fl-server << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $SERVICE_USER $SERVICE_USER
    sharedscripts
    postrotate
        systemctl reload secure-fl-server > /dev/null 2>&1 || true
    endscript
}
EOF

# Create environment file
print_status "Creating environment configuration..."
cat > "$PROJECT_DIR/.env" << EOF
# Secure FL Server Configuration
FL_SERVER_ADDRESS=0.0.0.0:8080
FL_NUM_ROUNDS=10
FL_MIN_CLIENTS=2
FL_SIMILARITY_THRESHOLD=0.5
FL_USE_TLS=true
FL_CERT_PATH=$CERT_DIR/server-cert.pem
FL_KEY_PATH=$CERT_DIR/server-key.pem
FL_CA_PATH=$CERT_DIR/ca-cert.pem
FL_LOG_DIR=$LOG_DIR
EOF

chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/.env"
chmod 600 "$PROJECT_DIR/.env"

# Display setup information
print_status "Setup completed successfully!"
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo "Project Directory: $PROJECT_DIR"
echo "Service User: $SERVICE_USER"
echo "Log Directory: $LOG_DIR"
echo "Certificate Directory: $CERT_DIR"
echo ""
echo "TLS Certificates:"
echo "  CA Certificate: $CERT_DIR/ca-cert.pem"
echo "  Server Certificate: $CERT_DIR/server-cert.pem"
echo "  Server Key: $CERT_DIR/server-key.pem"
echo ""
echo "To start the server:"
echo "  sudo systemctl start secure-fl-server"
echo "  # OR"
echo "  sudo supervisorctl start secure-fl-server"
echo ""
echo "To check server status:"
echo "  sudo systemctl status secure-fl-server"
echo "  # OR"
echo "  sudo supervisorctl status secure-fl-server"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/server.log"
echo ""
echo "To test health:"
echo "  $PROJECT_DIR/health_check.sh"
echo ""
echo "IMPORTANT:"
echo "  1. Copy your project files to $PROJECT_DIR"
echo "  2. Update security group to allow port 8080 from client IPs"
echo "  3. Distribute CA certificate to clients for mutual TLS"
echo "  4. Configure client encryption keys securely"
echo ""
print_warning "For production use, replace self-signed certificates with CA-signed certificates!"
echo "=========================================="
