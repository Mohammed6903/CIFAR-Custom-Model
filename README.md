# PyTorch CIFAR-10 Classifier: From Training to API Deployment

This project documents the complete lifecycle of a deep learning model, from building a custom Convolutional Neural Network (CNN) in PyTorch to deploying it as a live, interactive web API. The goal was to gain a hands-on, foundational understanding of the entire end-to-end machine learning pipeline.

The repository is structured into two main parts:
1.  **Training:** The Google Colab notebook (`training.ipynb`) used to train the CNN on the CIFAR-10 dataset.
2.  **Deployment:** The FastAPI application (`main.py`) that serves the trained model via a REST API.

## Core Learnings & Key Concepts Covered

This project was a practical exercise in understanding the following key areas:

### 1. Neural Network Fundamentals (PyTorch)
-   **Custom Architecture:** Built a simple CNN from scratch using `torch.nn.Module`, composing layers like `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, and `nn.Linear` to understand how data flows through a network.
-   **The Training Loop:** Implemented the complete training process manually, gaining a deep understanding of the five core steps:
    1.  **Forward Pass:** Getting predictions from the model.
    2.  **Loss Calculation:** Quantifying model error using `nn.CrossEntropyLoss`.
    3.  **Zeroing Gradients:** Resetting gradients before backpropagation.
    4.  **Backward Pass:** Calculating gradients with `loss.backward()`.
    5.  **Weight Update:** Adjusting model parameters with an optimizer like `Adam`.
-   **Data Handling:** Utilized `torchvision.transforms` for data normalization and `DataLoader` for efficient, batched data pipelines.

### 2. Model Deployment as a Service
-   **API Development with FastAPI:** Built a robust, production-ready web server to expose the model's prediction capabilities.
-   **Inference Mode:** Learned the importance of `model.eval()` and `torch.no_grad()` to ensure the model behaves correctly and efficiently during prediction (disabling dropout, batch norm updates, and gradient calculations).
-   **Data Serialization:** Handled the process of receiving an image as a file over HTTP, converting it from bytes to a PIL Image, and finally transforming it into the exact PyTorch Tensor format the model expects.
-   **Dependency Management:** Understood the need for a `requirements.txt` file to ensure the application runs consistently in any environment.

### 3. Bridging Theory and Practice
-   **Model Persistence:** Learned to save a trained model's state (`model.state_dict()`) and load it for inference, decoupling the training environment from the deployment environment.
-   **Real-World Constraints:** Addressed practical challenges like ensuring input images (e.g., PNGs with alpha channels) are correctly converted to the RGB format the model was trained on.

## Project Structure

```
.
├── model/
│   └── cifar10_cnn.pth      # The saved, trained model weights
├── main.py                  # The FastAPI application for deployment
├── training.ipynb           # The Google Colab notebook for training
├── requirements.txt         # Python dependencies for the API
└── README.md                # You are here!
```

## How to Run the Deployed API Locally

This guide assumes you have Python 3.8+ installed.

### 1. Clone the Repository & Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Run the Server

With the `model/cifar10_cnn.pth` file in place, run the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The server will start, typically on `http://127.0.0.1:8000`.

### 3. Test the API

FastAPI provides automatic, interactive documentation.

1.  Open your web browser and navigate to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.
2.  Expand the `POST /predict` endpoint.
3.  Click "Try it out".
4.  Click "Choose File" and upload an image of a car, dog, airplane, etc.
5.  Click "Execute".

You will see the model's prediction and confidence score returned as a JSON response.
