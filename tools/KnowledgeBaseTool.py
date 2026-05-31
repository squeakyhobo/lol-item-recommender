import json
import os
from smolagents import Tool

class KnowledgeBaseTool(Tool):
    name = "consult_manuals"
    description = "Searches the Yorick/Top Lane knowledge base. Contains Wave Control, Top Lane Guides, and CHAMPION MATCHUP TIPS."
    inputs = {}
    output_type = "string"

    def forward(self):
        base_dir = r"C:\Users\Lucas\Desktop\LTA\data"
        # Now reading FOUR sources of truth
        files = ["wave_rules.json", "top_lane_guide.json", "pro_benchmarks.json", "yorick_matchups.json"]
        
        combined_knowledge = {}
        for filename in files:
            file_path = os.path.join(base_dir, filename)
            try:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        combined_knowledge[filename] = json.load(f)
            except Exception as e:
                combined_knowledge[filename] = f"Error reading: {str(e)}"
        
        return json.dumps(combined_knowledge, indent=2)
