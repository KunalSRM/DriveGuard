# DriveGuard
Distracted Driver detection project
Driver Guard – Real-Time Driver Monitoring System
📌 Overview

Driver Guard is a deep learning–based project that detects and classifies driver activities in real time using computer vision.
The system helps improve road safety by identifying unsafe behaviors (e.g., using phone, drinking, drowsiness) and providing alerts.

🛠 Features

Real-time driver activity detection using webcam or video input.

Image preprocessing and feature extraction using CNN.

Classification into predefined categories (e.g., safe driving, phone usage, drowsiness, etc.).

Metadata integration with Excel for activity labeling and evaluation.

📂 Project Structure
DriverGuard/
│── dataset/              # Images (not uploaded due to size)
│── model/                # Saved trained model (.h5 / .pth)
│── train_driverguard.py  # Training script
│── detect_realtime.py    # Real-time detection script
│── requirements.txt      # Dependencies
│── README.md             # Project description

📊 Dataset

Dataset contains driver images with corresponding activity labels.

Each image is converted into a numerical tensor before feeding into the neural network.

Labels (from Excel) are nominal categorical data (e.g., "Safe Driving", "Phone Usage", etc.).

⚠️ Note: Due to dataset size, it is not uploaded here. You can download from [link placeholder].

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/DriverGuard.git
cd DriverGuard


Install dependencies:

pip install -r requirements.txt


Train the model:

python train_driverguard.py


Run real-time detection:

python detect_realtime.py

🧠 How It Works

Preprocessing – Images are resized & normalized to prepare input for CNN.

Feature Extraction – CNN layers learn important patterns (face, eyes, phone, etc.).

Classification – Fully connected layers classify into predefined driver states.

Real-Time Detection – Model runs continuously on live camera feed.

🚀 Future Improvements

Add sound-based alerts for unsafe behavior.

Optimize for mobile deployment.

Support larger, more diverse datasets.

🙌 Acknowledgments

Dataset: [Your dataset source]

Frameworks: TensorFlow / PyTorch, OpenCV
