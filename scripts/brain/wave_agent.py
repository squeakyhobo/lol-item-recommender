import os
import sys
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

# Add project root to path so we can find 'tools'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.WaveControlGuideTool import WaveGuideTool

# 1. Load your GitHub Token from .env
# Make sure you have GITHUB_TOKEN=your_token_here in your .env file
load_dotenv(override=True)
github_token = os.getenv("GITHUB_TOKEN")

def create_wave_coach():
    """
    Creates a SmolAgent equipped with the Yorick Wave Manual.
    Uses GitHub Models (Student Package) for reasoning.
    """
    if not github_token:
        print("[!] Error: GITHUB_TOKEN not found in .env file.")
        return None

    # 2. Define the Model
    # GitHub Student provides access to high-end models for free.
    # We point to the GitHub Models endpoint using LiteLLM.
    model = LiteLLMModel(
        model_id="openai/gpt-4o", # GitHub Models are OpenAI-compatible
        api_key=github_token,
        api_base="https://models.inference.ai.azure.com" # GitHub Models endpoint
    )

    # 3. Create the Agent
    # We give it our manual WaveGuideTool so it can 'read' our rules.
    agent = CodeAgent(
        tools=[WaveGuideTool()], 
        model=model,
        add_base_tools=False, # We want it strictly focused on our game rules
        max_steps=3 # Prevents the AI from looping too long
    )
    
    return agent

if __name__ == "__main__":
    # Test the coach
    coach = create_wave_coach()
    if coach:
        print("\n[*] Yorick Wave Coach is active. Testing situational reasoning...")
        
        test_query = """
        Current State: I am Yorick (Level 3). 
        Enemy: Darius (Level 3). 
        Wave State: Even, in the middle of the lane. 
        Ghouls: 0. 
        Graves: 1.
        
        Question: What should I do with the wave right now?
        """
        
        response = coach.run(test_query)
        print(f"\n--- COACH ADVICE ---\n{response}")
