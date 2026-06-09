import httpx
from openai import OpenAI
from chutes_e2ee import ChutesE2EETransport

API_KEY = "cpk_025a7a164b89407e9b06446ebc11d909.10d6e259c9f3552693f400639a72ceae.hgc9ipRpevqcOplWGbmtmHHhahioHP05"
API_BASE = "https://api.lorebary.com/chutes/"

# MODEL_SLUG = "chutes-google-gemma-3-31b-turbo-tee"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE+"/v1"
)

response = client.chat.completions.create(
    model="google/gemma-4-31B-turbo-TEE",
    messages=[
        {"role":
             "user",
         "content":
             "Выведи Hello world"}
    ],
)

print(response.choices[0].message.content)