import tensorflow as tf
from paper_trader.models.model_loader import directional_loss


def convert_models(symbol: str):
    """Load legacy model and resave with proper serialization."""
    model = tf.keras.models.load_model(f"models/lstm/{symbol}_window_X.keras", compile=False)
    model.compile(optimizer='adam', loss=directional_loss, metrics=['mae'])
    model.save(f"models/lstm/{symbol}_window_X_converted.keras")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_models.py SYMBOL")
    else:
        convert_models(sys.argv[1])
