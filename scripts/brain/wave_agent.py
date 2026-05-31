import os
import sys
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

# Add project root to path
BASE_DIR = r"C:\Users\Lucas\Desktop\LTA"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tools.KnowledgeBaseTool import KnowledgeBaseTool

# Load environment
load_dotenv(override=True)

def create_wave_coach():
    """
    Creates a Local Yorick Coach using Ollama via LiteLLM.
    Equipped with a full Top Lane Knowledge Base.
    """
    print("[*] Coach: Booting LOCAL intelligence (Qwen 3.5 via Ollama)...")
    
    try:
        model = LiteLLMModel(
            model_id="ollama/qwen3.5:9b", 
            api_base="http://localhost:11434"
        )
        
        # Build the Agent with the new multi-file KnowledgeBaseTool
        agent = CodeAgent(
            tools=[KnowledgeBaseTool()], 
            model=model,
            add_base_tools=False,
            max_steps=3
        )
        return agent
    except Exception as e:
        print(f"[!] Local Coach Failed: {e}")
        return None

if __name__ == "__main__":
    coach = create_wave_coach()
    if coach:
        test_query = "It is minute 3.4. I am level 3 Yorick vs Darius. Wave is even. What should I be looking out for?"
        print(f"\n[*] Asking Local Qwen...")
        print(f"\nCOACH ADVICE: {coach.run(test_query)}")
