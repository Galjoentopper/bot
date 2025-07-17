from datetime import datetime

class PredictionTrendAnalyzer:
    def __init__(self, lookback_periods: int = 5):
        self.lookback_periods = lookback_periods
        self.prediction_history = {}

    def add_prediction(self, symbol: str, prediction: dict):
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        self.prediction_history[symbol].append({
            'timestamp': datetime.now(),
            'prediction': prediction
        })
        if len(self.prediction_history[symbol]) > self.lookback_periods:
            self.prediction_history[symbol] = self.prediction_history[symbol][-self.lookback_periods:]

    def is_trend_deteriorating(self, symbol: str) -> bool:
        if symbol not in self.prediction_history:
            return False
        recent_predictions = self.prediction_history[symbol][-3:]
        if len(recent_predictions) >= 2:
            latest = recent_predictions[-1]['prediction']
            previous = recent_predictions[-2]['prediction']
            confidence_declining = latest.get('confidence', 0) < previous.get('confidence', 0)
            direction_negative = latest.get('direction') == 'DOWN'
            return confidence_declining and direction_negative
        return False
