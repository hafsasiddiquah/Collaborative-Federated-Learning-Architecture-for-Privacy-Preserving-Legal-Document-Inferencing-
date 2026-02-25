#!/bin/bash
#
# Complete AWS Deployment Script for Secure Federated Learning
# Deploys the entire FL system to AWS EC2 with Docker containers
#

set -e

# Configuration
PROJECT_NAME="secure-legal-fl"
AWS_REGION="us-east-1"
INSTANCE_TYPE="t3.medium"
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS
KEY_NAME="secure-fl-key"
SECURITY_GROUP_NAME="${PROJECT_NAME}-sg"
IAM_ROLE_NAME="${PROJECT_NAME}-role"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi

    print_status "Prerequisites check passed"
}

# Create IAM role and policies
create_iam_role() {
    print_header "Creating IAM Role and Policies"

    # Create IAM policy for EC2
    cat > /tmp/ec2-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeImages",
                "ec2:DescribeKeyPairs",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeSubnets",
                "ec2:DescribeVpcs",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create the policy
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "${PROJECT_NAME}-policy" \
        --policy-document file:///tmp/ec2-policy.json \
        --query 'Policy.Arn' \
        --output text 2>/dev/null || \
        aws iam get-policy \
        --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/${PROJECT_NAME}-policy" \
        --query 'Policy.Arn' \
        --output text)

    print_status "Created IAM policy: $POLICY_ARN"

    # Create trust policy for EC2
    cat > /tmp/trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    # Create IAM role
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --query 'Role.Arn' \
        --output text || print_warning "IAM role may already exist"

    # Attach policy to role
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn "$POLICY_ARN"

    # Create instance profile
    aws iam create-instance-profile \
        --instance-profile-name "${PROJECT_NAME}-profile" || print_warning "Instance profile may already exist"

    aws iam add-role-to-instance-profile \
        --instance-profile-name "${PROJECT_NAME}-profile" \
        --role-name "$IAM_ROLE_NAME" || print_warning "Role may already be attached"

    print_status "IAM role and instance profile created"
}

# Create security group
create_security_group() {
    print_header "Creating Security Group"

    # Create security group
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Security group for Secure FL system" \
        --query 'GroupId' \
        --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

    print_status "Security group created: $SG_ID"

    # Add inbound rules
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --no-cli-pager || print_warning "SSH rule may already exist"

    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --no-cli-pager || print_warning "API port rule may already exist"

    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 8080 \
        --cidr 0.0.0.0/0 \
        --no-cli-pager || print_warning "FL server port rule may already exist"

    print_status "Security group rules configured"
}

# Launch EC2 instance
launch_ec2_instance() {
    print_header "Launching EC2 Instance"

    # Get the latest Ubuntu AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)

    print_status "Using AMI: $AMI_ID"

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile Name="${PROJECT_NAME}-profile" \
        --user-data file://aws/ec2_setup.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-server},{Key=Project,Value=${PROJECT_NAME}}]" \
        --query 'Instances[0].InstanceId' \
        --output text)

    print_status "EC2 instance launched: $INSTANCE_ID"

    # Wait for instance to be running
    print_status "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    print_status "Instance public IP: $PUBLIC_IP"

    # Wait for SSH to be available
    print_status "Waiting for SSH to be available..."
    for i in {1..30}; do
        if nc -z "$PUBLIC_IP" 22 2>/dev/null; then
            break
        fi
        sleep 10
    done

    echo "$INSTANCE_ID" > .instance_id
    echo "$PUBLIC_IP" > .public_ip
}

# Build and deploy Docker containers
deploy_application() {
    print_header "Building and Deploying Application"

    INSTANCE_ID=$(cat .instance_id)
    PUBLIC_IP=$(cat .public_ip)

    # Copy project files to instance
    print_status "Copying project files to EC2 instance..."
    scp -o StrictHostKeyChecking=no -i "~/.ssh/${KEY_NAME}.pem" -r . ubuntu@"$PUBLIC_IP":~/ || {
        print_error "Failed to copy files. Make sure your key pair exists and is in ~/.ssh/"
        exit 1
    }

    # SSH into instance and deploy
    print_status "Deploying application on EC2 instance..."
    ssh -o StrictHostKeyChecking=no -i "~/.ssh/${KEY_NAME}.pem" ubuntu@"$PUBLIC_IP" << EOF
        cd secure_legal_fl

        # Build Docker images
        echo "Building Docker images..."
        docker-compose build

        # Start services
        echo "Starting services..."
        docker-compose up -d

        # Wait for services to be healthy
        echo "Waiting for services to be healthy..."
        sleep 30

        # Check service status
        docker-compose ps
        docker-compose logs --tail=20
EOF

    print_status "Application deployed successfully"
}

# Setup monitoring and logging
setup_monitoring() {
    print_header "Setting up Monitoring and Logging"

    INSTANCE_ID=$(cat .instance_id)
    PUBLIC_IP=$(cat .public_ip)

    # Create CloudWatch log group
    aws logs create-log-group \
        --log-group-name "/aws/ec2/${PROJECT_NAME}" || print_warning "Log group may already exist"

    print_status "CloudWatch logging configured"
}

# Main deployment function
main() {
    print_header "AWS Deployment for Secure Federated Learning"

    check_prerequisites
    create_iam_role
    create_security_group
    launch_ec2_instance
    deploy_application
    setup_monitoring

    PUBLIC_IP=$(cat .public_ip)

    print_header "Deployment Complete!"
    echo -e "${GREEN}Secure FL System deployed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  API Server:     http://$PUBLIC_IP:8000"
    echo -e "  FL Server:      http://$PUBLIC_IP:8080"
    echo -e "  Health Check:   http://$PUBLIC_IP:8000/health"
    echo ""
    echo -e "${BLUE}SSH Access:${NC}"
    echo -e "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "  1. Configure your clients to connect to the FL server"
    echo -e "  2. Upload your trained model for inference"
    echo -e "  3. Monitor logs: docker-compose logs -f"
}

# Handle command line arguments
case "\${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup_resources
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: \$0 [deploy|cleanup|status]"
        exit 1
        ;;
esac