# pip install openai-agents
# pip install python-dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# Gemini API setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model selection
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator Agent
translator = Agent(
    name='Translator Agent',
    instructions="""
    You are a translation agent. Translate any English sentence into proper Urdu. 
    Only reply with the translated text in Urdu. Do not explain anything.
    """
)

# Input sentence (you can change this)
english_text = "Thank you Sir Ali Jawad for your inspiring teaching and constant support. Grateful to have you as our teacher!"

# Run translation
response = Runner.run_sync(
    translator,
    input=english_text,
    run_config=config
)

# Show output in terminal
print("\n🟢 Urdu Translation:")
print(response.final_output)

# Save to file
with open("urdu_output.txt", "w", encoding="utf-8") as f:
    f.write(response.final_output)
