import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// 1. Define the Custom Normalization Layer with trainable weight

// 2. Custom Hook to Load the Model with proper error handling
const useModel = (modelUrl: string) => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoading(true);
        console.log("Loading model from:", modelUrl);

        // Load the model with custom objects
        const loadedModel = await tf.loadLayersModel(modelUrl);

        const inputShape = [1, 55, 47, 3]; // Example, update with the correct shape of your model's input
        loadedModel.build(inputShape);

        // Debug and initialize layers

        await loadedModel.compile({
          optimizer: "adam",
          loss: "meanSquaredError",
          metrics: ["accuracy"],
        });

        setModel(loadedModel);
        console.log("Model loaded successfully.");
        loadedModel.summary();
      } catch (err: any) {
        console.error("Error loading model:", err);
        setError(`Model failed to load: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadModel();

    // Cleanup function
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [modelUrl]);

  return { model, loading, error };
};

// 3. Main Component with better error handling and memory management
const ModelLoader: React.FC = () => {
  const modelUrl = "/web_model/model.json";
  const { model, loading, error } = useModel(modelUrl);
  const [prediction, setPrediction] = useState<number[] | null>(null);

  const handlePrediction = async () => {
    if (!model) return;

    try {
      // Perform the prediction and handle async operations outside of tf.tidy
      const predArray = await tf.tidy(() => {
        const dummyInput = tf.randomNormal([1, 55, 47, 3]); // Adjust shape as needed for your model
        const predictionTensor = model.predict(dummyInput) as tf.Tensor;

        // Dispose tensors synchronously
        dummyInput.dispose();
        predictionTensor.dispose();

        // Return the tensor from tf.tidy (no async logic here)
        return predictionTensor;
      });

      // Wait for the tensor to be converted to an array (outside of tf.tidy)
      const arrayResult = await predArray.array();

      // Safely update state with the first prediction array
      if (Array.isArray(arrayResult) && Array.isArray(arrayResult[0])) {
        setPrediction(arrayResult[0] as number[]); // Ensure it's a number[]
      } else {
        setPrediction([]); // Fallback to an empty array if shape doesn't match
      }
    } catch (err: any) {
      console.error("Error during prediction:", err);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">TensorFlow.js Model Loader</h1>

      {loading && (
        <div className="bg-blue-100 p-4 rounded">
          <p>Loading model...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-100 p-4 rounded">
          <p className="text-red-700">Error: {error}</p>
        </div>
      )}

      {!loading && !error && model && (
        <div className="space-y-4">
          <p className="text-green-600">Model loaded successfully!</p>
          <button
            onClick={handlePrediction}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Run Prediction
          </button>

          {prediction && (
            <div className="mt-4">
              <h2 className="text-xl font-semibold">Prediction Result:</h2>
              <pre className="bg-gray-100 p-4 rounded mt-2">
                {JSON.stringify(prediction, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelLoader;
