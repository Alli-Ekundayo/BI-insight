import json
import os
import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class InsightLogger:
    def __init__(self):
        self.log_file = config.BASE_DIR / "data/insight_log.json"
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
                
    def log_insight(self, insight: dict):
        """
        FRD: "Maintain a log of past insights and suggestions... for historical tracking."
        """
        insight['timestamp'] = datetime.datetime.now().isoformat()
        
        try:
            with open(self.log_file, 'r+') as f:
                data = json.load(f)
                data.append(insight)
                f.seek(0)
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error logging insight: {e}")

    def get_past_insights(self):
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
