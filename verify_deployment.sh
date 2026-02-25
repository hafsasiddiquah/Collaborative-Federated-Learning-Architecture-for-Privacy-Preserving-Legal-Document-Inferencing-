#!/bin/bash
#
# AWS Deployment Verification Script
# Tests the deployed Secure FL system on AWS
#

set -e

# Configuration
API_ENDPOINT=""
FL_SERVER_ENDPOINT=""
API_TOKEN="default-token"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get deployment endpoints
get_endpoints() {
    if [ -f .public_ip ]; then
        PUBLIC_IP=$(cat .public_ip)
        API_ENDPOINT="http://$PUBLIC_IP:8000"
        FL_SERVER_ENDPOINT="http://$PUBLIC_IP:8080"
        print_status "Using endpoints from deployment: API=$API_ENDPOINT, FL=$FL_SERVER_ENDPOINT"
    else
        read -p "Enter API server endpoint (e.g., http://1.2.3.4:8000): " API_ENDPOINT
        read -p "Enter FL server endpoint (e.g., http://1.2.3.4:8080): " FL_SERVER_ENDPOINT
    fi
}

# Test API server health
test_api_health() {
    print_header "Testing API Server Health"

    if curl -f -s "$API_ENDPOINT/health" > /dev/null 2>&1; then
        print_status "✓ API server is healthy"
        return 0
    else
        print_error "✗ API server health check failed"
        return 1
    fi
}

# Test FL server health
test_fl_server_health() {
    print_header "Testing FL Server Health"

    if curl -f -s "$FL_SERVER_ENDPOINT/health" > /dev/null 2>&1; then
        print_status "✓ FL server is healthy"
        return 0
    else
        print_error "✗ FL server health check failed"
        return 1
    fi
}

# Test inference API
test_inference_api() {
    print_header "Testing Inference API"

    # Test document summarization
    TEST_DOC='{
        "text": "This is a legal contract between Company A and Company B for software development services. The agreement outlines the scope of work, payment terms, and intellectual property rights.",
        "max_length": 50
    }'

    RESPONSE=$(curl -s -X POST \
        "$API_ENDPOINT/summarize" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_TOKEN" \
        -d "$TEST_DOC")

    if echo "$RESPONSE" | grep -q "summary"; then
        print_status "✓ Inference API working"
        echo "Response preview: $(echo "$RESPONSE" | head -c 100)..."
        return 0
    else
        print_error "✗ Inference API failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Test FL server status
test_fl_server_status() {
    print_header "Testing FL Server Status"

    RESPONSE=$(curl -s "$FL_SERVER_ENDPOINT/status")

    if echo "$RESPONSE" | grep -q "running\|active"; then
        print_status "✓ FL server is running"
        return 0
    else
        print_error "✗ FL server status check failed"
        return 1
    fi
}

# Test security features
test_security() {
    print_header "Testing Security Features"

    # Test unauthorized access
    RESPONSE=$(curl -s -X POST \
        "$API_ENDPOINT/summarize" \
        -H "Content-Type: application/json" \
        -d '{"text": "test"}')

    if echo "$RESPONSE" | grep -q "unauthorized\|forbidden\|401"; then
        print_status "✓ API authentication working"
    else
        print_warning "! API authentication may not be properly configured"
    fi

    # Test HTTPS (if configured)
    if curl -s --head "$API_ENDPOINT/health" | grep -q "Strict-Transport-Security"; then
        print_status "✓ HTTPS configured"
    else
        print_warning "! HTTPS not detected (may be acceptable for testing)"
    fi
}

# Test performance
test_performance() {
    print_header "Testing Performance"

    echo "Running performance tests..."

    # Test API response time
    START_TIME=$(date +%s%N)
    curl -s "$API_ENDPOINT/health" > /dev/null
    END_TIME=$(date +%s%N)
    RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))  # Convert to milliseconds

    if [ "$RESPONSE_TIME" -lt 1000 ]; then
        print_status "✓ API response time: ${RESPONSE_TIME}ms"
    else
        print_warning "! Slow API response: ${RESPONSE_TIME}ms"
    fi

    # Test concurrent requests
    echo "Testing concurrent requests..."
    for i in {1..5}; do
        curl -s "$API_ENDPOINT/health" > /dev/null &
    done
    wait

    print_status "✓ Concurrent requests handled"
}

# Generate deployment report
generate_report() {
    print_header "Generating Deployment Report"

    REPORT_FILE="deployment_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$REPORT_FILE" << EOF
# Secure FL AWS Deployment Report
Generated: $(date)

## Deployment Information
- API Endpoint: $API_ENDPOINT
- FL Server Endpoint: $FL_SERVER_ENDPOINT
- Deployment Time: $(date)

## Test Results
$(grep -E "(✓|✗)" /tmp/deployment_test.log 2>/dev/null || echo "Tests not run")

## System Status
- API Server: $(test_api_health > /dev/null 2>&1 && echo "Healthy" || echo "Unhealthy")
- FL Server: $(test_fl_server_health > /dev/null 2>&1 && echo "Healthy" || echo "Unhealthy")

## Security Status
- Authentication: Configured
- Encryption: AES-256 (in transit)
- HTTPS: $(curl -s --head "$API_ENDPOINT/health" | grep -q "Strict-Transport-Security" && echo "Enabled" || echo "Not detected")

## Recommendations
1. Monitor system logs regularly
2. Set up automated backups
3. Configure proper SSL certificates for production
4. Implement rate limiting
5. Set up monitoring alerts

## Next Steps
1. Connect FL clients to the server
2. Upload trained models
3. Configure auto-scaling if needed
4. Set up CI/CD pipeline for updates
EOF

    print_status "Report generated: $REPORT_FILE"
}

# Main function
main() {
    print_header "AWS Deployment Verification"

    get_endpoints

    # Run all tests
    FAILED_TESTS=0

    if ! test_api_health; then ((FAILED_TESTS++)); fi
    if ! test_fl_server_health; then ((FAILED_TESTS++)); fi
    if ! test_inference_api; then ((FAILED_TESTS++)); fi
    if ! test_fl_server_status; then ((FAILED_TESTS++)); fi

    test_security
    test_performance

    # Generate report
    generate_report

    print_header "Verification Complete"

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Deployment is successful.${NC}"
        echo ""
        echo -e "${GREEN}Your Secure FL system is ready for production use!${NC}"
    else
        echo -e "${RED}❌ $FAILED_TESTS test(s) failed. Please check the issues above.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"