# pip install google-generativeai
import google.generativeai as genai

# Configure the client with your API key
genai.configure(api_key="AIzaSyCHz71YCqqSe7gYf_4hYr5Nu7izCjou7ms")

# Specify the model to use
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Prompt the user for input
user_prompt = "What is GenAI?"

# Generate a response using the Gemini API
response = model.generate_content(user_prompt)

# Print the response
print("Generated text:\n", response.text)
