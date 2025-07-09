
# 🎓 Suspicious Activity Detection in Exam using Deep Learning

This project uses deep learning and computer vision to detect suspicious or cheating behavior during online or offline exams. The system processes video input to identify and classify actions as either **"normal"** or **"suspicious"**, helping proctors maintain exam integrity.

---

## 📌 Features

- 🔍 Real-time suspicious behavior detection
- 🧠 Deep Learning-based classification using CNN
- 🎥 Frame-wise analysis using OpenCV
- 🖼️ Automatic face detection using Haar Cascades
- 🖥️ GUI interface for easy video input and results visualization
- 📝 Timestamped log of detected suspicious actions

---

## 🧠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy, PIL**
- **Tkinter** (for GUI)

---

## 📂 Project Structure

```
SuspiciousActivityDetection/
│
├── Train_FDD_cnn.py        # CNN training script
├── show_FDD_video.py       # Real-time video detection script
├── model.h5                # Trained CNN model
├── gui.py                  # GUI interface
├── dataset/                # Labeled training images (suspicious/normal)
├── frames/                 # Temporary frames extracted from video
└── README.md               # This file
```

---

## 🏗️ How It Works

1. **Frame Extraction**: Video is split into individual frames using OpenCV.
2. **Face Detection**: Each frame is scanned for faces using Haar Cascades.
3. **CNN Prediction**: Detected faces are fed into a Convolutional Neural Network trained to classify suspicious behavior.
4. **Display**: Results (label, timestamp, bounding boxes) are shown in the GUI.
5. **Logging**: Suspicious events are logged with frame number and timestamp.

---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/amarak24/Suspicious-Activity-Detection-in-Exam-Using-Deep-Learning.git

```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the GUI
```bash
python GUI_main.py
```

---

## 🧪 Model Training

To retrain the model:

1. Place labeled images inside `dataset/normal` and `dataset/suspicious`
2. Run the training script:
```bash
python Train_FDD_cnn.py
```
3. This generates a new `model.h5` file.

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **Confusion Matrix**
- **Classification Report**

The model achieves ~90% accuracy on test data.

---

## 🖼️ Sample Output

- 🕒 Timestamp: 00:02:15  
- 🎞️ Frame: 875  
- 🧍‍♂️ Activity: *Suspicious*  
- 📦 Action: Displayed bounding box on the video feed  

---


## 💡 Future Improvements

- Integrate audio analysis for verbal cues
- Support for multiple students in one video
- Alert system with real-time notifications
- Deploy as a web application

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
