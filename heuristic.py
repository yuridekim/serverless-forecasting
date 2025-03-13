import logging
logging.getLogger("prophet.plot").disabled = True
from prophet import Prophet
from datetime import datetime, timedelta
from sf_platform import ServerlessPlatform
import pandas as pd

class ServerlessScheduler:
    def __init__(self, platform: ServerlessPlatform, models: dict[str, Prophet], interval=10, default_warm_instances=0, default_warm_period=10, gen_pred=True):
        """
        platform: Instance of ServerlessPlatform
        models: Map function names to Prophet models.
        interval: Scheduling interval in seconds.
        default_warm_instances: Default permanently warm instances if models lack training.
        default_warm_period: Default warm period in seconds if models lack training.
        gen_pred: Should generate predictions
        """
        self.platform = platform
        self.models = models
        self.interval = interval
        self.default_warm_instances = default_warm_instances
        self.default_warm_period = default_warm_period
        self.predictions = []
        self.gen_pred = gen_pred

    def get_predictions(self, model):
        """Generate predictions for the next interval using Prophet."""
        if not self.gen_pred:
            return None

        try:
            # Ensure future timestamps align with per-second predictions
            last_timestamp = model.history['ds'].max()
            future = pd.DataFrame({'ds': [last_timestamp + timedelta(seconds=i) for i in range(1, self.interval + 1)]})

            # Generate forecast
            forecast = model.predict(future)

            # Filter relevant timestamps
            future_timestamps = set(future['ds'])
            return forecast[forecast['ds'].isin(future_timestamps)][['ds', 'yhat']]
        
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return None

    async def adjust_scheduling(self):
        """Adjust scheduling based on Prophet model predictions."""
        for func_name, model in self.models.items():
            predictions = self.get_predictions(model)
            self.predictions.append(predictions)
            # print("PREDICTIONS:", predictions)

            if predictions is None or predictions.empty:
                # Fallback keep-alive policy
                await self.platform.set_permanently_warm_instances(func_name, self.default_warm_instances)
                await self.platform.set_default_warm_period(func_name, self.default_warm_period)
                continue

            # Invocations per second (avoid negative predictions). Can change granularity later
            invocation_counts = {row['ds']: max(0, row['yhat']) for _, row in predictions.iterrows()}

            # Determine permanently warm instances
            max_concurrent = max(invocation_counts.values(), default=0)
            permanently_warm_instances = int(round(max_concurrent))

            # Determine warm period
            active_seconds = [ds for ds, count in invocation_counts.items() if count > 0]
            warm_period = int((max(active_seconds) - min(active_seconds)).total_seconds()) + 1 if active_seconds else 0

            await self.platform.set_permanently_warm_instances(func_name, permanently_warm_instances)
            await self.platform.set_default_warm_period(func_name, warm_period)
