# lambda_functions.py
import json
import boto3
import base64

# Initialize the S3 client
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """Function to serialize target data from S3"""
    # Get the S3 address from the Step Function event input
    key    = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from S3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read the data from the file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body' : json.dumps({
            "image_data" : image_data,
            "s3_bucket"  : bucket,
            "s3_key"     : key,
            "inferences" : []
        })
    }

# Note: For image classification function
ENDPOINT = "image-classification-2024-08-14-15-44-26-432"

def lambda_handler(event, context):
    """Function to invoke a SageMaker endpoint for image classification"""
    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Make a prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = ENDPOINT,
        ContentType  = 'image/png',
        Body         = image
    )

    # Decode the response from the endpoint
    inferences_str = response['Body'].read().decode('utf-8')

    # Convert the inferences JSON string to a Python list
    inferences = json.loads(inferences_str)
    inferences = [float(x) for x in inferences]

    # Return the inference result as part of the event
    event["inferences"] = inferences
    return {
        'statusCode' : 200,
        'body'       : json.dumps(event)
    }

# Note: For threshold check function
THRESHOLD = 0.8

def lambda_handler(event, context):
    """Function to check if any inference meets the threshold"""
    # Grab the inferences from the event
    inferences = event['inferences']

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) >= THRESHOLD

    # If our threshold is met, pass our data back out of the Step Function
    # Else, end the Step Function with an error
    if meets_threshold:
        return {
            'statusCode' : 200,
            'body'       : json.dumps(event)
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
