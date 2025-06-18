from groq import Groq

def call_groq_model(groq_key, query, llm_model):
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        model=llm_model,
    )
    return response