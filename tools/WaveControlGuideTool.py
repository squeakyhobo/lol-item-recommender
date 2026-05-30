import json
import os
from smolagents import Tool

class WaveGuideTool(Tool):
    name = "get_wave_rules"
    description = "Searches the Yorick and General Top Lane Wave Control manual. Use this to find the correct strategy based on current minion count and champion state."
    inputs = {}
    output_type = "string"

    def forward(self):
        # We use absolute paths to ensure the agent can find the data folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "wave_rules.json")
        
        try:
            with open(file_path, "r") as f:
                rules = json.load(f)
            return json.dumps(rules, indent=2)
        except Exception as e:
            return f"Error reading wave manual: {str(e)}"
