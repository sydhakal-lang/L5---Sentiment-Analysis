import requests
from config import HF_API_KEY

api_url = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

text = input("Enter your text:")

response = requests.post(
    api_url,
    headers=headers,
    json={"inputs":text}
)

if response.status_code == 200:
    results = response.json()

    print("Raw response:", results)

    sentiment = results[0][0]["label"]
    score = results[0][0]["score"]

    print(f"Sentiment: {sentiment}, confidence: {score:.4f}")
else:
    print(f"Error {response.status_code}: {response.text}")
