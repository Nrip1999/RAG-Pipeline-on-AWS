import boto3
import botocore.config
import json

from datetime import datetime

def blog_generate_using_bedrock(blogtopic:str)-> str:
    prompt=f"""<s>[INST]Human: Write a completed article on the topic {blogtopic}
    Assistant:[/INST]
    """

    body={
        "prompt":prompt,
        "max_gen_len":1024, #Specify the maximum number of tokens to use in the generated response. The model truncates the response once the generated text exceeds max_gen_len.
        "temperature":0.5, #Use a lower value to decrease randomness in the response.
        "top_p":0.9 #Use a lower value to ignore less probable options. Set to 0 or 1.0 to disable.
    }

    try:
        bedrock_client=boto3.client("bedrock-runtime",region_name="ap-south-1",
                             config=botocore.config.Config(read_timeout=300,retries={'max_attempts':3}))
        response=bedrock_client.invoke_model(body=json.dumps(body),modelId="meta.llama3-8b-instruct-v1:0")

        response_content=response.get('body').read()
        response_data=json.loads(response_content)
        print(response_data)
        blog_details=response_data['generation']
        return blog_details
    except Exception as e:
        print(f"Error generating the blog:{e}")
        return ""

def save_blog_details_s3(s3_key,s3_bucket,generate_blog):
    s3=boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body =generate_blog )
        print("Code saved to s3")

    except Exception as e:
        print("Error when saving the code to s3")



def lambda_handler(event, context):
    # TODO implement
    event=json.loads(event['body'])
    blogtopic=event['Article']

    generate_blog=blog_generate_using_bedrock(blogtopic=blogtopic)

    if generate_blog:
        current_time=datetime.now().strftime('%H%M%S')
        s3_key=f"blog-output/{current_time}.txt"
        s3_bucket='host-content-1'
        save_blog_details_s3(s3_key,s3_bucket,generate_blog)


    else:
        print("No article was generated")

    return{
        'statusCode':200,
        'body':json.dumps('Article generation is completed')
    }