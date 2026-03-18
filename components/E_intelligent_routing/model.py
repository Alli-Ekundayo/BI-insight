from .features import RoutingFeatureEngine
import pandas as pd

class SmartRouter:
    def __init__(self):
        self.engine = RoutingFeatureEngine()
        self.is_initialized = False

    def train(self, df: pd.DataFrame):
        """
        Calculates the internal routing tables from historically loaded data.
        """
        print("Initializing Smart Router aggregation tables...")
        self.engine.compute_routing_stats(df)
        self.is_initialized = True
        print(f"Smart Router ready. Processed {len(self.engine.stats)} routing triplets.")

    def recommend(self, issuer_id: str, scheme: str, alpha: float = 0.7):
        if not self.is_initialized:
            return {"error": "Router not initialized. Call train(df) first."}
            
        return self.engine.get_best_acquirer(issuer_id, scheme, alpha)
