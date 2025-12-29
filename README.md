# Neural Navigator 🧠🗺️

A Multi-Modal Neural Network acting as a "Smart GPS." This model fuses Computer Vision (CNN) and Natural Language Processing (LSTM) to predict a navigation path on a 2D map based on visual cues and text commands.

## 📂 Project Structure
```
.
├── data_loader.py      # Custom Dataset class for loading Images, JSONs, and Tokenizing text
├── model.py            # Multi-Modal Architecture (CNN + LSTM + Fusion + Decoder)
├── train.py            # Training loop with MSE Loss and Adam Optimizer
├── predict.py          # Inference script to visualize predicted paths
├── requirements.txt    # Dependencies
├── assignment_dataset  # Folder that contains the datasets to be trained 
  ├── data
  └── test_data
└── navigator_model.pth # Saved model weights (generated after training)
```
Make sure to have this folder structure.

# 🚀 Quick Start

## 1. Github Cloning
Clone the following github repositories into your workspace folder.

```Bash
git clone https://github.com/AiSaurabhPatil/assignment_dataset.git
git clone https://github.com/ProEvolution03/ros-neural-navigator.git
```

## 2. Installation
Ensure you have Python 3.10 or 3.11 installed.
Then run the following command.

```Bash
pip install -r requirements.txt
```

## 3. Training
Train the model on the dataset. This will save navigator_model.pth.

```Bash

python train.py
```

## 4. Prediction & Visualization
Run the model on test images to see the generated path.

```Bash

python predict.py
```


# 🛠️ Challenges & Solutions
## 1. What was the hardest part of this assignment?
The most challenging aspect was designing the Fusion Layer effectively. We are combining two fundamentally different data modalities: a static 2D image and a sequential text command. The challenge was ensuring that the "Vision" features and the "Language" features were represented in a compatible latent space before concatenation.

Initial thought: Flatten all inputs and feed them to a Dense layer.

Realization: This approach destroyed the spatial and sequential nature of the data too early in the network.

Solution: I standardized both encoders to output a fixed feature vector of size hidden_dim (64). For the text, I extracted the final hidden state of the LSTM. For the image, I projected the CNN output using a Linear layer. This ensured that when concatenated, neither modality dominated the other due to mismatched vector magnitudes.

## 2. Describe a specific bug you encountered and how you debugged it.
The Bug: Tensor Shape Mismatch at the Fusion Step. RuntimeError: Sizes of tensors must match except in dimension 1. Got [32, 64] and [32, 1, 64]

Debugging Process:

Identification: The error occurred immediately during the first forward pass in model.py at the line: torch.cat((img_feats, text_feats), dim=1).

Hypothesis: I suspected that the LSTM text encoder was outputting a 3D tensor (Batch, Sequence, Features) or (Batch, 1, Features) instead of a flat 2D tensor like the image features.

Tools: I inserted "print debugging" statements inside the forward method:

```Python

print(f"Image shape: {img_feats.shape}")
print(f"Text shape: {text_feats.shape}")
```

Verification: The logs showed Text shape: torch.Size([32, 1, 64]). The LSTM returns (batch, seq_len, hidden) even if we only want the last state.

The Fix: I applied .squeeze(0) to the hidden state output from the LSTM to remove the unnecessary dimension before concatenation.

Code change: text_feats = text_hidden.squeeze(0)

## 3. Accuracy achieved on the test dataset.
Since this is a Regression Task (predicting continuous X, Y coordinates) rather than a classification task, "Accuracy" in percentage is not the standard metric. We evaluated performance using Mean Squared Error (MSE) Loss.

Final Training Loss (MSE): 0.0025 (Converged from initial loss of ~0.3)

Visual Accuracy: In visual tests on the validation set, the model successfully directs the path to the correct target shape/color in approximately 95% of cases, with the path endpoints landing within the target's bounding box.
