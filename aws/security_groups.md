# AWS Security Group Configuration Guide
## Secure Federated Learning Infrastructure

This document outlines the security group configurations for deploying the Secure Federated Learning system on AWS.

## Security Groups Overview

### 1. FL Server Security Group (`secure-fl-server-sg`)

**Purpose**: Controls access to the federated learning aggregation server running on EC2.

**Inbound Rules**:
```
Type: SSH (22)
Source: Your IP/0.0.0.0/0 (restrict in production)
Description: SSH access for administration

Type: Custom TCP (8080)
Source: Client IPs/0.0.0.0/0 (restrict to known client IPs)
Description: Federated Learning server port

Type: HTTPS (443)
Source: 0.0.0.0/0 or specific IPs
Description: API access (if using reverse proxy)

Type: HTTP (80)
Source: 0.0.0.0/0 or specific IPs
Description: Health checks and API access
```

**Outbound Rules**:
```
Type: All traffic
Destination: 0.0.0.0/0
Description: Allow all outbound traffic
```

### 2. API Server Security Group (`secure-fl-api-sg`)

**Purpose**: Controls access to the FastAPI inference server.

**Inbound Rules**:
```
Type: SSH (22)
Source: Your IP/0.0.0.0/0 (restrict in production)
Description: SSH access for administration

Type: Custom TCP (8000)
Source: Application IPs/0.0.0.0/0
Description: FastAPI server port

Type: HTTPS (443)
Source: 0.0.0.0/0 or specific IPs
Description: API access via reverse proxy

Type: HTTP (80)
Source: 0.0.0.0/0 or specific IPs
Description: Health checks
```

**Outbound Rules**:
```
Type: All traffic
Destination: 0.0.0.0/0
Description: Allow all outbound traffic
```

### 3. Client Instances Security Group (`secure-fl-client-sg`)

**Purpose**: Security group for federated learning client instances.

**Inbound Rules**:
```
Type: SSH (22)
Source: Your IP/0.0.0.0/0 (restrict in production)
Description: SSH access for administration
```

**Outbound Rules**:
```
Type: Custom TCP (8080)
Destination: FL Server Security Group
Description: Connect to FL aggregation server

Type: HTTPS (443)
Destination: 0.0.0.0/0
Description: Download models and datasets

Type: All traffic
Destination: 0.0.0.0/0
Description: General outbound access
```

## Security Best Practices

### 1. Principle of Least Privilege
- Restrict SSH access to specific IP ranges
- Use separate security groups for different components
- Regularly audit and update security group rules

### 2. Network Segmentation
- Place FL server in private subnet if possible
- Use NAT gateway for outbound internet access
- Implement VPC endpoints for AWS services

### 3. Monitoring and Logging
- Enable VPC Flow Logs
- Use AWS Config for security group change monitoring
- Set up CloudWatch alarms for security events

### 4. Encryption in Transit
- All FL communication uses TLS 1.3
- API endpoints use HTTPS
- SSH access uses key-based authentication

## Terraform Configuration Example

```hcl
# FL Server Security Group
resource "aws_security_group" "fl_server" {
  name_prefix = "secure-fl-server-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_client_cidrs
    description = "FL server port"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_cidrs
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "secure-fl-server-sg"
  }
}

# API Server Security Group
resource "aws_security_group" "api_server" {
  name_prefix = "secure-fl-api-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_api_cidrs
    description = "FastAPI server port"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_cidrs
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "secure-fl-api-sg"
  }
}
```

## CloudWatch Monitoring

### Security Group Metrics to Monitor
- Number of security group changes
- Unauthorized access attempts
- Port scanning activities

### Sample CloudWatch Alarm
```json
{
  "AlarmName": "SecurityGroupChanges",
  "AlarmDescription": "Alert on security group changes",
  "MetricName": "SecurityGroupChanges",
  "Namespace": "AWS/Config",
  "Statistic": "Sum",
  "ComparisonOperator": "GreaterThanThreshold",
  "Threshold": 0,
  "Period": 300,
  "EvaluationPeriods": 1
}
```

## Compliance Considerations

### HIPAA (Healthcare Data)
- Implement end-to-end encryption
- Use private subnets for sensitive workloads
- Enable VPC Flow Logs for audit trails

### GDPR (EU Data Protection)
- Minimize data collection and retention
- Implement proper access controls
- Document data processing activities

### SOC 2
- Regular security assessments
- Change management procedures
- Incident response planning

## Incident Response

### Security Group Compromise Response
1. **Immediate Actions**:
   - Revoke compromised credentials
   - Update security groups to block malicious IPs
   - Enable enhanced monitoring

2. **Investigation**:
   - Review CloudTrail logs
   - Analyze VPC Flow Logs
   - Check CloudWatch metrics

3. **Recovery**:
   - Rotate access keys
   - Update security groups
   - Implement additional controls

## Maintenance

### Regular Security Reviews
- Monthly security group audit
- Quarterly access review
- Annual penetration testing

### Automation
- Use AWS Config Rules for compliance monitoring
- Implement automatic remediation for common issues
- Regular backup of security configurations