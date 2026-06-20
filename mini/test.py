import json
import os
from google import genai
from dotenv import load_dotenv
load_dotenv() 


# client = genai.Client(api_key=os.environ["GEMINI_API"])
# chat = client.chats.create(model="gemini-3.5-flash")

# # ... (Imagine your conversation loops here) ...
# chat.send_message("Hi, how are you? I like apples")


# saved_history = []
# for message in chat.get_history():
#     # Gather text from all parts of the message
#     text_content = "".join([part.text for part in message.parts if part.text])
    
#     saved_history.append({
#         "role": message.role,
#         "text": text_content
#     })

# # Save to a local JSON file
# with open("chat_session.json", "w", encoding="utf-8") as f:
#     json.dump(saved_history, f, indent=4, ensure_ascii=False)

# print("Chat history successfully saved!")





import os
import json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API"])

# ---- HOW TO LOAD ----
history_to_load = []

if os.path.exists("chat_session.json"):
    with open("chat_session.json", "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    for msg in saved_data:
        # Map raw data back to explicit SDK history objects
        part = types.Part.from_text(text=msg["text"])
        if msg["role"] == "user":
            history_to_load.append(types.UserContent(parts=[part]))
        else:
            history_to_load.append(types.ModelContent(parts=[part]))

# Re-initialize the chat session with the loaded history array
chat = client.chats.create(
    model="gemini-3.5-flash",
    history=history_to_load
)

# Test if it remembers context from the last session
response = chat.send_message("Remind me what fruit we were talking about earlier?")
print(response.text)
# Output: We were talking about apples!