# Character-Level RNN from Scratch (NumPy)

This project is a minimal implementation of a **character-level Recurrent Neural Network (RNN)** built entirely from scratch using **NumPy**. It demonstrates how sequence models learn patterns in text data by predicting the next character in a sequence.

---

## 📌 Overview

The model is trained on a tiny dataset:

```
"hello"
```

It learns to predict the next character given a sequence:

| Input | Target |
| ----- | ------ |
| h     | e      |
| e     | l      |
| l     | l      |
| l     | o      |

After training, the model can predict the next character (e.g., `"hell" → "o"`).

---

## ⚙️ Features

* Character-level tokenization
* One-hot encoding
* Manual forward pass (RNN)
* Backpropagation Through Time (BPTT)
* Gradient clipping (to prevent exploding gradients)
* Simple training loop
* Next-character prediction

---

## 🧠 How It Works

### 1. Data Preprocessing

* Extract unique characters from the dataset
* Create mappings:

  * `char_to_ix`: character → index
  * `ix_to_char`: index → character

---

### 2. One-Hot Encoding

Each character is converted into a one-hot vector of size `vocab_size`.

Example:

```
e → [0, 1, 0, 0]^T
```

---

### 3. Model Architecture

The RNN consists of:

* **Input → Hidden weights (`Wxh`)**
* **Hidden → Hidden weights (`Whh`)**
* **Hidden → Output weights (`Why`)**
* **Bias terms (`bh`, `by`)**

Hidden state update:

```
h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
```

Output:

```
y_t = Why * h_t + by
p_t = softmax(y_t)
```

---

### 4. Forward Pass

* Iterates through input characters
* Updates hidden state at each step
* Produces probability distribution over next characters

---

### 5. Loss Function

Cross-entropy loss:

```
Loss = -log(p(correct_char))
```

---

### 6. Backpropagation (BPTT)

* Gradients are computed backwards through time
* Updates:

  * `dWxh`, `dWhh`, `dWhy`
  * `dbh`, `dby`

Gradient clipping:

```
np.clip(grad, -5, 5)
```

---

### 7. Training

* Uses simple gradient descent
* Learning rate: `0.1`
* Runs for `500 iterations`

---

## 🚀 Usage

### Run the script:

```bash
python rnn.py
```

### Output:

* Training loss every 100 iterations
* Final prediction

Example:

```
iteration 0 loss: ...
iteration 100 loss: ...
...
Prediction: o
```

---

## 🔮 Prediction

Use:

```python
predict("hell")
```

Expected output:

```
o
```

---

## 📊 Key Concepts Demonstrated

* Recurrent Neural Networks (RNNs)
* Sequence modeling
* Backpropagation Through Time (BPTT)
* Softmax and cross-entropy loss
* Gradient clipping

---

## ⚠️ Limitations

* Extremely small dataset ("hello")
* No batching (single sequence training)
* No optimization tricks (Adam, RMSProp, etc.)
* Vanilla RNN (no LSTM/GRU → prone to vanishing gradients)

---

## 💡 Possible Improvements

* Train on larger text datasets
* Replace RNN with LSTM/GRU
* Add sampling to generate text sequences
* Use better optimizers (Adam)
* Implement mini-batch training

---

## 📚 Learning Purpose

This project is ideal for:

* Beginners learning RNN internals
* Understanding how deep learning works under the hood
* Practicing NumPy-based neural network implementation

---

## 🏁 Summary

This is a simple but powerful demonstration of how neural networks can learn sequential patterns—even from scratch. While minimal, it captures the core mechanics behind modern language models.

---
