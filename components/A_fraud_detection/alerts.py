import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class AlertDispatcher:
    def __init__(self):
        # Slack/Email settings would be here
        pass
        
    def get_template(self, alert_type: str, category: str = "GENERAL"):
        """
        Returns (prompt, full_details) based on the FRD specifications.
        """
        templates = {
            "VOLUME": {
                "GENERAL": {
                    "prompt": "Attention! Unusual Transaction Volume Anomaly Detected - Immediate Investigation Advised.",
                    "full": (
                        "Suspicious Transaction Alert: Our system has detected an anomaly in your recent transaction activity. "
                        "This could be indicative of a potentially fraudulent transaction. We recommend you review this transaction "
                        "for further investigation to ensure the security of your accounts and funds. If you confirm any "
                        "unauthorised activity, please take immediate action to secure your accounts."
                    )
                },
                "TERMINALS": {
                    "prompt": "Important Notice: Unusual Transaction Volume Detected at POS Terminals - Urgent Attention Required",
                    "full": "We would like to bring to your attention that our monitoring system has detected an anomaly in a recent transaction at a POS terminal. This anomaly may indicate a potential issue related to fraud or operational concerns."
                },
                "CARDS": {
                    "prompt": "Critical Alert: Unusual Transaction Volume Detected with ATM Cards – Immediate Action Needed!",
                    "full": "We would like to bring to your attention that our monitoring system has detected an anomaly in a recent transaction involving an ATM card. This anomaly may indicate a potential issue related to fraud or operational concerns."
                },
                "CHANNELS": {
                    "prompt": "Urgent Alert: Unusual Transaction Volume Anomaly with Channel – Immediate Action Required!",
                    "full": "We would like to bring to your attention that our monitoring system has detected an anomaly in a recent transaction on a specific channel. This anomaly may indicate a potential issue related to fraud or operational concerns."
                }
            },
            "VALUE": {
                "GENERAL": {
                    "prompt": "Attention! Unusual Transaction Value Anomaly Detected - Immediate Investigation Advised.",
                    "full": "We would like to bring to your attention that our monitoring system has detected an anomaly in a recent transaction value. This anomaly may indicate a potential issue related to fraud or operational concerns."
                }
            }
        }
        return templates.get(alert_type, {}).get(category, templates["VOLUME"]["GENERAL"])

    def dispatch(self, scored_txn: dict, alert_type: str = "VOLUME", category: str = "GENERAL"):
        """
        Formats and prints/sends the alert using FRD templates.
        """
        template = self.get_template(alert_type, category)
        
        prompt = template['prompt']
        description = template['full']
        
        # Snapshot Details
        txn_id = scored_txn.get('transaction_id', 'N/A')
        ts = scored_txn.get('transaction_time', 'N/A')
        amt = scored_txn.get('transaction_amount', 0)
        terminal = scored_txn.get('terminal_id', 'N/A')
        pan = scored_txn.get('pan', 'N/A')
        acc = scored_txn.get('merchant_id', 'N/A') # Assuming merchant_id as "Customer Account" context
        
        recommendation = "We recommend further investigation into this transaction to ensure the security and integrity of your financial operations."
        
        # Format for display
        print("\n" + "!" * 60)
        print(f"PROMPT: {prompt}")
        print("-" * 60)
        print(f"TITLE: {alert_type} Anomaly Detected ({category})")
        print(f"DESCRIPTION: {description}")
        print("\nDATA SNAPSHOT:")
        print(f"  - Transaction ID: {txn_id}")
        print(f"  - Date and Time: {ts}")
        print(f"  - Amount/Volume: {amt}")
        if category == "TERMINALS": print(f"  - Terminal ID: {terminal}")
        if category == "CARDS": print(f"  - PAN: {pan}")
        print(f"  - Account/Merchant: {acc}")
        print(f"\nRECOMMENDATION: {recommendation}")
        print("!" * 60 + "\n")
