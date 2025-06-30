from groq import Groq
import instructor

def call_groq_model(groq_key, query, llm_model):
    client = Groq(api_key=groq_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a commonsense knowledge engineer. Your task is to provide accurate commonsense knowledge. Return **ONLY** valid JSON.",
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        model=llm_model,
    )

    return chat_completion.choices[0].message.content

def call_groq_model_with_instructor(groq_key, responsemodel, query, llm_model):
    client = instructor.from_groq(Groq(api_key=groq_key), mode=instructor.Mode.JSON)
    response = client.chat.completions.create(
        model=llm_model,
        response_model=responsemodel, # Specify the response model
        messages=[
            {"role": "system", "content": "Your job is to provide concept information for the given text."},
            {"role": "user", "content": query}
        ],
        temperature=0.65,)
    return response