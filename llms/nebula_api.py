import requests

def call_nebula_model(key, prompt, model):
    url = 'https://nebula.cs.vu.nl/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a commonsense knowledge engineer. Your task is to provide accurate commonsense knowledge. Return **ONLY** valid JSON.",
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    return response.json()['choices'][0]['message']['content']