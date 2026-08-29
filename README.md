# Heart Disease Prediction App 💓

An interactive Machine Learning web application built using **Streamlit** and **Scikit-Learn** (KNN Classifier) to predict heart disease risk based on clinical features.

## 🚀 Features

- Interactive input form for medical metrics (Age, Blood Pressure, Cholesterol, ECG, Max HR, etc.).
- Machine Learning classification powered by K-Nearest Neighbors (`KNeighborsClassifier`).
- Instant risk prediction output (**High Risk** or **Low Risk**).

## 📁 Repository Structure

- `app.py`: Streamlit web application frontend and prediction logic.
- `HeartdiseaseFinal.ipynb`: Jupyter notebook detailing model training, data cleaning, and evaluation.
- `heart.csv`: Heart disease dataset.
- `knn_heart_model.pkl`: Saved KNN Classifier model artifact.
- `heart_scaler.pkl`: StandardScaler transformer artifact.
- `heart_columns.pkl`: List of feature column names expected by the model.
- `requirements.txt`: Python package dependencies.

## 🛠️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Machine-Learning-Part-3
   ```

2. **Create and activate a Python 3.13 virtual environment**:
   ```bash
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
