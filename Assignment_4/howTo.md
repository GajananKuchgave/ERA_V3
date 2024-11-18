# MNIST CNN Training Visualization

This project implements a 4-layer CNN trained on MNIST with real-time training visualization.

## Setup

1. Install required packages: 
bash
pip install torch torchvision flask numpy matplotlib

2. Project structure:
mnist_cnn/
├── howTo.md
├── train.py
├── model.py
├── templates/
│ └── index.html
└── static/
└── style.css

3. Run the training:
bash
python train.py

4. Open your browser and navigate to:
http://localhost:5000