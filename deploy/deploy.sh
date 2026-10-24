#!/usr/bin/env bash
# Deploys Spotter to AWS Lambda (zip package + public Function URL).
# Run it in AWS CloudShell (it already has the AWS CLI, Python and pip):
#   git clone https://github.com/DarkModeGirl22/spotter && cd spotter && bash deploy/deploy.sh
# Safe to run again: it updates the existing function.
set -euo pipefail

REGION="${AWS_REGION:-eu-north-1}"
NAME="spotter"
ROLE="spotter-lambda-role"
MODEL_ID="${SPOTTER_MODEL_ID:-}"            # optional Bedrock model / inference profile id
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="spotter-deploy-${ACCOUNT}-${REGION}"
BUILD="$(mktemp -d)"

echo "==> 1/6 Building package (OpenCV 5 for Lambda Python 3.13)"
python3 -m pip install -q --target "$BUILD/pkg" \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
  --implementation cp --python-version 3.13 --only-binary=:all: \
  opencv-python-headless==5.0.0.93 numpy==2.5.3 fastapi==0.162.0 starlette==1.7.0 \
  pydantic==2.13.5 mangum==0.22.0
# keep under Lambda's 250 MB limit (boto3 is already in the Lambda runtime)
find "$BUILD/pkg" -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
find "$BUILD/pkg" -name "*.pyi" -delete
rm -rf "$BUILD/pkg/numpy/f2py" "$BUILD/pkg/numpy/_pyinstaller"
cp -r server web models "$BUILD/pkg/"
(cd "$BUILD/pkg" && zip -qr9 "$BUILD/spotter.zip" .)
echo "    package: $(du -sh "$BUILD/pkg" | cut -f1) unzipped, $(du -sh "$BUILD/spotter.zip" | cut -f1) zipped"

echo "==> 2/6 Uploading to S3 ($BUCKET)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
fi
aws s3 cp "$BUILD/spotter.zip" "s3://$BUCKET/spotter.zip" --region "$REGION" --only-show-errors

echo "==> 3/6 IAM role (logs + Polly + Bedrock only)"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10   # let IAM propagate
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name spotter-ai --policy-document '{
  "Version":"2012-10-17","Statement":[
    {"Effect":"Allow","Action":["polly:SynthesizeSpeech"],"Resource":"*"},
    {"Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":"*"}]}'
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

ENV="Variables={SPOTTER_MODEL_ID=$MODEL_ID,SPOTTER_BEDROCK_REGION=$REGION,SPOTTER_POLLY_REGION=eu-west-1,SPOTTER_VOICE_CACHE=/tmp/spotter-voice}"

echo "==> 4/6 Lambda function"
if aws lambda get-function --function-name "$NAME" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --region "$REGION" \
    --s3-bucket "$BUCKET" --s3-key spotter.zip >/dev/null
  aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"
  aws lambda update-function-configuration --function-name "$NAME" --region "$REGION" \
    --environment "$ENV" >/dev/null
else
  for i in 1 2 3 4 5; do   # a brand-new role can take a few seconds to be usable
    if aws lambda create-function --function-name "$NAME" --region "$REGION" \
        --runtime python3.13 --architectures x86_64 --handler server.lambda_handler.handler \
        --role "$ROLE_ARN" --code S3Bucket="$BUCKET",S3Key=spotter.zip \
        --memory-size 2048 --timeout 30 --environment "$ENV" >/dev/null; then break; fi
    echo "    role not ready yet, retrying in 10 s"; sleep 10
  done
fi
aws lambda wait function-active-v2 --function-name "$NAME" --region "$REGION"
aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"
aws logs put-retention-policy --log-group-name "/aws/lambda/$NAME" --retention-in-days 14 \
  --region "$REGION" 2>/dev/null || true

echo "==> 5/6 Public HTTPS link (Function URL)"
URL="$(aws lambda get-function-url-config --function-name "$NAME" --region "$REGION" \
        --query FunctionUrl --output text 2>/dev/null || true)"
if [ -z "$URL" ] || [ "$URL" = "None" ]; then
  URL="$(aws lambda create-function-url-config --function-name "$NAME" --region "$REGION" \
          --auth-type NONE --query FunctionUrl --output text)"
fi
aws lambda add-permission --function-name "$NAME" --region "$REGION" --statement-id url-public \
  --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null 2>&1 || true
aws lambda add-permission --function-name "$NAME" --region "$REGION" --statement-id url-invoke \
  --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true

echo "==> 6/6 Checking it answers"
sleep 3
curl -s --max-time 60 "${URL}api/health" || true
echo
echo "============================================================"
echo " Spotter is live: $URL"
echo "============================================================"
rm -rf "$BUILD"
