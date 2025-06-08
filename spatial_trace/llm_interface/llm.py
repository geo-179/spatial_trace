import os
from openai import OpenAI

# The OpenAI client automatically looks for the OPENAI_API_KEY environment variable.
# If you don't set it as an environment variable, you can pass it directly:
client = OpenAI(api_key="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")
# client = OpenAI()

try:
    # This is the basic API call to the GPT-4o model
    response = client.chat.completions.create(
        model="gpt-4o",  # Specifies the GPT-4o model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello! How are you?"
            }
        ],
        max_tokens=150,  # Optional: The maximum number of tokens to generate
        temperature=0.7    # Optional: Controls the randomness of the output
    )

    # Extracting and printing the main content of the response
    assistant_response = response.choices[0].message.content
    print("GPT-4o:", assistant_response)

except Exception as e:
    print(f"An error occurred: {e}")
