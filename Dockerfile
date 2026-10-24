# AWS Lambda container image: OpenCV 5 pose pipeline + coach agent + web app
FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY server ${LAMBDA_TASK_ROOT}/server
COPY web ${LAMBDA_TASK_ROOT}/web
COPY models ${LAMBDA_TASK_ROOT}/models

ENV SPOTTER_VOICE_CACHE=/tmp/spotter-voice
CMD ["server.lambda_handler.handler"]
