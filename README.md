# 🧠 NeuroMood

A neural network that classifies EEG brainwave signals into emotional states: **Positive**, **Negative**, and **Neutral**.

This project was built as my first neural network using **PyTorch**, inspired by the digit recognition task in Michael Nielsen's "Neural Networks and Deep Learning", but applied to a more unique and real-world dataset: EEG emotion signals.

---

## 📊 Dataset (Kaggle)

This project uses the [EEG Signals for Emotion Recognition](https://www.kaggle.com/datasets/muhammadkhalil01/eeg-signals-emotion-classification) dataset from Kaggle.

It contains EEG features extracted from brainwave signals while subjects experienced different emotions. The task is to classify the emotional state into:

- `"POSITIVE"`
- `"NEGATIVE"`
- `"NEUTRAL"`

### 📁 Dataset Details

- CSV format with **2548 numerical features per row**
- One label column indicating the emotion
- Well-balanced between the 3 classes

### ⚙️ Preprocessing

- `LabelEncoder` used to convert string labels to integer class indices
- Features scaled using `StandardScaler` to improve neural network training
- Data converted into PyTorch tensors and loaded using `DataLoader` for batching

---

## 🧠 Model Architecture

The model is a simple fully connected feedforward neural network built using PyTorch:

```python
nn.Sequential(
    nn.Linear(2548, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

```
📈 Results
✅ Final Test Accuracy: ~98%
After training on the EEG dataset for 10 epochs with scaled features, the model achieved extremely high performance on the test set.

🔍 Confusion Matrix
markdown
Copy
Edit
               Predicted
               NEG  NEU  POS
Actual  NEG    139    0    3
        NEU      0  143    0
        POS      5    1  136
(NEG = Negative, NEU = Neutral, POS = Positive)

✅ NEUTRAL class was predicted perfectly.

✅ Very low misclassifications: just 9 out of 427 samples total.

📋 Classification Report
markdown
Copy
Edit
              precision    recall  f1-score   support

    NEGATIVE       0.97      0.98      0.98       142
     NEUTRAL       0.99      1.00      0.99       143
    POSITIVE       0.98      0.96      0.97       142

    accuracy                           0.98       427
   macro avg       0.98      0.98      0.98       427
weighted avg       0.98      0.98      0.98       427


## 🧰 Tech Stack

| Tool / Library     | Purpose                            |
|--------------------|------------------------------------|
| **Python 3.10+**   | Core programming language          |
| **PyTorch**        | Deep learning framework            |
| **NumPy**          | Numerical computation              |
| **scikit-learn**   | Preprocessing & evaluation         |
| **Matplotlib**     | Plotting graphs                    |
| **Seaborn**        | Confusion matrix heatmaps          |
| **Jupyter Notebook** | Interactive training + debugging |

---

## 🧪 Future Directions

- 🧬 Try LSTM or 1D CNN for modeling EEG signals as sequences
- 🧠 Integrate live EEG data from Muse/OpenBCI devices
- 📊 Visualize neuron activations per emotion
- 🌐 Deploy as a browser-based demo for real-time feedback
- 🎯 Apply model explainability tools like SHAP or Grad-CAM

---

## ✍️ Author

Made by [Gurmehar Singh Aujla](https://github.com/gurmeharsingh)  
Built during early deep learning exploration using PyTorch  
Inspired by Michael Nielsen’s *Neural Networks and Deep Learning* 📘

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, remix, and improve it!

> ⭐ *If you found this project useful, feel free to star it and share your thoughts!*  

