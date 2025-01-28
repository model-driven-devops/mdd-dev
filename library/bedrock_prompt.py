#!/usr/bin/python

from __future__ import (absolute_import, division, print_function)
__metaclass__ = type

DOCUMENTATION = r'''
---
module: bedrock_prompt
short_description: Send a prompt to AWS Bedrock and return the response
version_added: "1.0"
description:
    - This module allows you to send a prompt to AWS Bedrock for processing and retrieve the response, supporting both Text Completion and Messages API.
options:
    prompt:
        description: The text prompt to send to Bedrock.
        required: true
        type: str
    model_id:
        description: The ID of the Bedrock model to use for processing the prompt.
        required: true
        type: str
    region:
        description: AWS region where Bedrock is deployed.
        required: true
        type: str
    max_tokens:
        description: The maximum number of tokens to generate in the response.
        required: false
        type: int
        default: 300
author:
    - Your Name
'''

EXAMPLES = r'''
- name: Send a prompt to Bedrock using Messages API with longer response
  bedrock_prompt:
    prompt: "What are the key benefits of cloud computing?"
    model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
    region: "us-west-2"
    max_tokens: 1000
'''

RETURN = r'''
bedrock_response:
    description: The response from AWS Bedrock.
    type: str
    returned: success
    sample: "Cloud computing offers scalability, flexibility, and cost efficiency."
'''

from ansible.module_utils.basic import AnsibleModule
import boto3
import json

def run_module():
    module_args = dict(
        prompt=dict(type='str', required=True),
        model_id=dict(type='str', required=True),
        region=dict(type='str', required=True),
        max_tokens=dict(type='int', required=False, default=300)
    )

    result = dict(
        changed=False,
        bedrock_response=''
    )

    module = AnsibleModule(
        argument_spec=module_args,
        supports_check_mode=True
    )

    if module.check_mode:
        module.exit_json(**result)

    prompt = module.params['prompt']
    model_id = module.params['model_id']
    region = module.params['region']
    max_tokens = module.params['max_tokens']

    try:
        bedrock = boto3.client(service_name='bedrock-runtime', region_name=region)
        
        # Determine which API to use based on model ID
        if "claude-3-sonnet" in model_id or "claude-3-5-sonnet" in model_id:
            # Use Messages API for models like Claude 3 Sonnet
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
            response = bedrock.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response['body'].read())
            result['bedrock_response'] = response_body.get('content', [{}])[0].get('text', 'No response from Bedrock')
        else:
            # Default to Text Completion API
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.5,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"]
            })
            response = bedrock.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response['body'].read())
            if 'completion' not in response_body:
                if 'output' in response_body and 'text' in response_body['output']:
                    result['bedrock_response'] = response_body['output']['text']
                else:
                    result['bedrock_response'] = f"No valid response found. Raw response: {json.dumps(response_body, indent=2)}"
            else:
                result['bedrock_response'] = response_body['completion']

        module.exit_json(**result)
    except Exception as e:
        module.fail_json(msg=str(e), **result)

def main():
    run_module()

if __name__ == '__main__':
    main()