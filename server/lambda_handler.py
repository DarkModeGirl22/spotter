"""AWS Lambda entry point (container image, behind a Lambda Function URL)."""
from mangum import Mangum

from .app import app

handler = Mangum(app, lifespan="off")
