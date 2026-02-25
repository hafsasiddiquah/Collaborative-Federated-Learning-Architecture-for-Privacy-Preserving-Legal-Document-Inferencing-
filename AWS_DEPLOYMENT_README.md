# AWS Deployment Guide for Secure Federated Learning

This guide provides step-by-step instructions for deploying the Secure Federated Learning system to AWS EC2.

## Prerequisites

1. **AWS Account**: Active AWS account with appropriate permissions
2. **AWS CLI**: Installed and configured
   ```bash
   pip install awscli
   aws configure
   ```
3. **Docker**: Installed locally for testing
4. **SSH Key Pair**: EC2 key pair for instance access

## Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# Make scripts executable
chmod +x deploy_aws.sh verify_deployment.sh

# Deploy to AWS
./deploy_aws.sh

# Verify deployment
./verify_deployment.sh
```

### Option 2: Manual Deployment

1. **Create EC2 Instance**
   ```bash
   aws ec2 run-instances \
     --image-id ami-0c7217cdde317cfec \
     --count 1 \
     --instance-type t3.medium \
     --key-name your-key-pair \
     --security-groups secure-legal-fl-sg
   ```

2. **Configure Security Groups**
   - SSH (22): 0.0.0.0/0 (restrict in production)
   - API (8000): 0.0.0.0/0
   - FL Server (8080): 0.0.0.0/0

3. **Deploy Application**
   ```bash
   # Copy files to instance
   scp -i your-key.pem -r . ubuntu@instance-ip:~

   # SSH into instance
   ssh -i your-key.pem ubuntu@instance-ip

   # Deploy
   cd secure_legal_fl
   docker-compose up -d
   ```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   FL Clients    │◄──►│   FL Server     │
│  (Edge Devices) │    │  (EC2 Instance) │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Inference API  │
                       │  (FastAPI)      │
                       └─────────────────┘
```

## Services

- **FL Server** (Port 8080): Federated learning coordination server
- **API Server** (Port 8000): Model inference and summarization API
- **Monitoring**: CloudWatch logs and health checks

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# API Configuration
API_TOKENS=your-secure-token-here

# Model Configuration
MODEL_PATH=meta-llama/Llama-2-7b-hf
```

### Security Considerations

1. **Network Security**:
   - Restrict security group rules to specific IP ranges
   - Use VPC with private subnets for production

2. **API Security**:
   - Use strong API tokens
   - Implement rate limiting
   - Enable HTTPS with SSL certificates

3. **Data Security**:
   - All communications use AES-256 encryption
   - Differential privacy protects against data leakage
   - Poisoning detection prevents malicious updates

## Monitoring and Maintenance

### Health Checks

```bash
# Check API health
curl http://your-instance:8000/health

# Check FL server status
curl http://your-instance:8080/status
```

### Logs

```bash
# View application logs
docker-compose logs -f

# View CloudWatch logs
aws logs tail /aws/ec2/secure-legal-fl --follow
```

### Updates

```bash
# Update application
docker-compose pull
docker-compose up -d

# Update EC2 instance
sudo apt update && sudo apt upgrade
```

## Troubleshooting

### Common Issues

1. **Docker Build Failures**:
   ```bash
   # Clear Docker cache
   docker system prune -a

   # Rebuild images
   docker-compose build --no-cache
   ```

2. **Port Conflicts**:
   - Check if ports 8000/8080 are available
   - Modify docker-compose.yml if needed

3. **Memory Issues**:
   - Increase EC2 instance type for larger models
   - Use model quantization to reduce memory usage

4. **Network Issues**:
   - Verify security group rules
   - Check VPC configuration

### Performance Optimization

1. **Model Optimization**:
   - Use 4-bit quantization
   - Enable PEFT/LoRA for fine-tuning
   - Use model caching

2. **Instance Sizing**:
   - t3.medium: Basic testing (4GB RAM)
   - t3.large: Production with small models (8GB RAM)
   - g4dn.xlarge: GPU instance for better performance

## Cost Estimation

- **EC2 Instance**: ~$30/month (t3.medium, on-demand)
- **Data Transfer**: ~$0.09/GB
- **Storage**: ~$1/month (20GB EBS)
- **CloudWatch**: ~$5/month

**Total Estimated Cost**: ~$36/month for basic deployment

## Production Checklist

- [ ] SSL certificates configured
- [ ] Security groups restricted
- [ ] Monitoring and alerts set up
- [ ] Backups configured
- [ ] Auto-scaling configured (if needed)
- [ ] CI/CD pipeline set up
- [ ] Load balancer configured (if needed)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Docker and AWS documentation
3. Check application logs
4. Open an issue in the project repository