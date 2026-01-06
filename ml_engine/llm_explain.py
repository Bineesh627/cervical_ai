import boto3
import json
from botocore.exceptions import ClientError

def generate_explanation(features_dict: dict) -> str:
    if not features_dict:
        return "No specific risk factors were identified for this analysis."

    safe_features = {k: float(v) for k, v in features_dict.items()}

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful medical assistant explaining risk factors for cervical cancer.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Based on these risk factor values:
{json.dumps(safe_features, indent=2)}

Create a detailed bulleted list explaining ALL of the risk factors above. For EACH factor, you MUST provide:
1. An emoji at the start
2. The factor name in bold (e.g., **Age**)
3. A clear 2-3 sentence explanation of WHY this factor affects cervical cancer risk

Format each bullet point like this:
🕰️ **Age**: As you get older, the risk of developing cervical cancer increases. This is because the longer you are exposed to HPV, the more likely it is to cause cell abnormalities that can lead to cancer.

Do NOT say "top 3" or limit the list. Explain ALL factors provided.
End with a brief disclaimer about consulting healthcare professionals.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    body = {
        "prompt": prompt,
        "max_gen_len": 800,  # Increased for all factors
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )

        response = bedrock.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response["body"].read())
        return response_body.get("generation", "").strip()

    except ClientError as e:
        print(f"AWS Bedrock ClientError: {e}")
        return "AI explanation is currently unavailable."
    except Exception as e:
        print(f"General Error: {e}")
        return "An unexpected error occurred."
