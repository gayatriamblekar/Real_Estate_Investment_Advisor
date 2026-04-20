# 🏡 Real Estate Investment Advisor

An end-to-end **Machine Learning pipeline** that helps analyze real estate properties and predict **investment viability** and **future price growth**.
This project simulates a real-world housing scenario in India and provides insights through an interactive web dashboard.

---

## 📸 Screenshots

<p align="center">
  <img src="Screenshots/Screenshot 2026-04-20 154422.png" width="45%" />
  <img src="Screenshots/Screenshot 2026-04-20 154515.png" width="45%" height =100%/>
</p>

<p align="center">
  <b>Investment Predictor</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>EDA Dashboard</b>
</p>

---

## 🚀 Key Features

* 📊 **End-to-End ML Pipeline** (Data → Model → Deployment)
* 🧠 **Investment Classification Model** (Good vs Bad Investment)
* 💰 **Future Price Prediction (5 Years)**
* 📈 **Automated Exploratory Data Analysis (EDA)** with saved visualizations
* 📉 **Model Performance Tracking** with evaluation metrics
* 🌐 **Interactive Streamlit Web App**
* 📦 **Modular and Scalable Code Structure**

---

## ⚙️ Project Workflow

### 1️⃣ Data Generation

* Generates synthetic housing dataset (`india_housing_prices.csv`)
* Includes:

  * State, City, Locality
  * Price, Size, BHK
  * Amenities and more

---

### 2️⃣ Data Preprocessing & Feature Engineering

* Handles missing values & outliers
* Encodes categorical features
* Creates new features:

  * Price per SqFt
  * Property Age
  * School Density Score
* Scales numerical features
* Outputs:

  * `cleaned_data.csv`
  * Saved encoders & scalers

---

### 3️⃣ Exploratory Data Analysis (EDA)

* Generates up to **20 visual insights**
* Detects trends, correlations, and patterns
* Saves charts in:

```
eda_charts/
```

---

### 4️⃣ Model Training & Tracking

#### 🔹 Classification (Investment Viability)

* Predicts if a property is a **Good Investment**

#### 🔹 Regression (Future Price)

* Predicts **price after 5 years**

#### 🔹 MLflow Tracking

* Logs:

  * Hyperparameters
  * Accuracy, F1 Score
  * RMSE, MAE

#### 🔹 Model Output

* Best models saved in:

```
models/
```

---

### 5️⃣ Web Application (Streamlit)

#### 🟢 Investment Predictor

* Accepts user input
* Predicts:

  * Investment quality
  * Future ROI
* Displays feature importance

#### 🔵 EDA Dashboard

* Displays pre-generated charts
* No runtime computation needed

#### 🟣 Model Performance

* Shows:

  * Metrics comparison
  * Confusion matrices

---

## 🧠 Workflow Summary

```
Data Generation
      ↓
Preprocessing & Feature Engineering
      ↓
Exploratory Data Analysis (EDA)
      ↓
Model Training (Classification + Regression)
      ↓
Model Tracking (MLflow)
      ↓
Deployment (Streamlit App)
```

---

## 🛠️ Tech Stack

* **Programming:** Python
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Tracking:** MLflow
* **Deployment:** Streamlit

---

## 📁 Project Structure

```
real_estate_advisor/
├── generate_dataset.py
├── preprocessing.py
├── eda.py
├── train_models.py
├── app.py
├── models/
├── eda_charts/
├── screenshots/
├── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/real_estate_advisor.git
cd real_estate_advisor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the pipeline (optional step-by-step)

```bash
python generate_dataset.py
python preprocessing.py
python eda.py
python train_models.py
```

### 4️⃣ Launch the app

```bash
streamlit run app.py
```

---

## 📊 Example Use Case

* Compare properties across cities
* Evaluate investment potential
* Predict future property value
* Analyze market trends visually

---

## 🔮 Future Improvements

* 🌍 Real-world dataset integration
* 🗺️ Map-based visualization
* 🔐 User authentication system
* 📱 Mobile-friendly UI
* 🤖 Advanced ML models (XGBoost, Deep Learning)

---


## ⭐ If you found this useful

Give this repo a ⭐ and share your feedback!

