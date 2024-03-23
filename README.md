# Face Mask Detector (COVID-19 era)

## Intro
The project is based on RCNN approach to detect faces and masks in images, including association to the specific person that trained on.
There are three main components in the project:
- Face detection
- Mask detection
- Person association


## Model Architecture
- Input layer: Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation function
- Hidden layer: Max pooling layer with pool size 2x2
- Hidden layer: Convolutional layer with 64 filters, kernel size 3x3, and ReLU activation function
- Hidden layer: Max pooling layer with pool size 2x2
- Hidden layer: Convolutional layer with 128 filters, kernel size 3x3, and ReLU activation function
- Hidden layer: Max pooling layer with pool size 2x2
- Flatten layer
- Dense layer with 128 units and ReLU activation function
- Output layer: Dense layer with 1 unit and sigmoid activation function