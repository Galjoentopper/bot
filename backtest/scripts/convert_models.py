import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper_trader.models.model_loader import directional_loss


import os

def convert_models(symbol: str):
    """Load legacy models and resave with proper serialization."""
    model_dir = os.path.join('models', 'lstm')
    for filename in os.listdir(model_dir):
        if filename.startswith(symbol.lower() + '_window_') and filename.endswith('.keras'):
            model_path = os.path.join(model_dir, filename)
            try:
                print(f"Converting model: {model_path}")
                model = tf.keras.models.load_model(model_path, custom_objects={"directional_loss": directional_loss}, compile=False)
                
                # Save architecture separately as JSON
                model_json = model.to_json()
                arch_path = model_path.replace('.keras', '_architecture.json')
                with open(arch_path, 'w') as json_file:
                    json_file.write(model_json)
                
                # Save weights separately
                weights_path = model_path.replace('.keras', '_weights.h5')
                model.save_weights(weights_path)
                
                # Save full model with optimizer
                model.compile(optimizer='adam', loss=directional_loss, metrics=['mae'])
                new_model_path = model_path.replace('.keras', '_converted.keras')
                model.save(new_model_path, include_optimizer=True)
                print(f"Saved converted model to: {new_model_path}")
            except Exception as e:
                print(f"Failed to convert model {model_path}: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_models.py SYMBOL")
    else:
        convert_models(sys.argv[1])
