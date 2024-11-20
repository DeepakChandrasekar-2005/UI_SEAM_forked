import os
import gdown
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    Input,
    Add,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

# Define the DeepID model construction function
def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
) -> Model:
    myInput = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
    x = Dropout(rate=0.99, name="D1")(x)

    x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
    x = Dropout(rate=0.99, name="D2")(x)

    x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
    x = Dropout(rate=0.99, name="D3")(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name="fc11")(x1)

    x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name="fc12")(x2)

    y = Add()([fc11, fc12])
    y = Activation("relu", name="deepid")(y)

    model = Model(inputs=[myInput], outputs=y)

    # Download weights if not present
    weights_path = os.path.expanduser("~/.deepface/weights/deepid_keras_weights.h5")
    if not os.path.isfile(weights_path):
        print("Downloading model weights...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        gdown.download(url, weights_path, quiet=False)

    model.load_weights(weights_path)
    return model


# Main script
if __name__ == "__main__":
    # Load the model
    model = load_model()

    tf.keras.backend.clear_session()

    print("Model loaded successfully.")

    # Load and preprocess images
    model.save('deepid.h5')
