# 💧 COMPFEST 16 Kaggle Challenge: Predicting Groundwater Hardness with ANN

This project was developed for the **Data Science Academy selection** at **COMPFEST 16**, Universitas Indonesia. The challenge focused on predicting **groundwater hardness** in Mexico using supervised regression techniques. A custom-built **Artificial Neural Network (ANN)** was implemented and evaluated using **R² Score**.

---

## 🎯 Objective

To build a regression model that accurately predicts the **hardness** of groundwater based on chemical composition, including calcium, magnesium, alkalinity, and other relevant variables. The goal is to support environmental monitoring and ensure water safety for public health.

---

## 📦 Dataset

- **Source**: Provided by COMPFEST 16 Committee  
- **Features**: Chemical attributes (e.g., Calcium, Magnesium, Alkalinity, pH, etc.)  
- **Target Variable**: `Hardness` (continuous value)  
- **Test Format**: Submission required in CSV with `id` and predicted `Hardness`

---

## 🧠 Model Overview

### Model: Artificial Neural Network (ANN)

Built using the **TensorFlow / Keras Sequential API** with the following architecture:

```python
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Build model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=0)

# Predict
y_pred = model.predict(X_test_scaled)
