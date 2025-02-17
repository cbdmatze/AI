# pip install groq
from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")

# Specify the model to use
model = "llama-3.3-70b-versatile"

# System task
system_prompt = "You are a helpful assistant."

# User's request
user_prompt = "What is GenAI?"

# Generate a response using the Groq API
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# Display the generated text
print("Generated text:\n", response.choices[0].message.content)
