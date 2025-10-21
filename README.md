# 🍷 Wine Quality Prediction using Machine Learning

This project predicts the quality of wine based on its physicochemical properties using various machine learning techniques.  
It analyzes wine characteristics such as acidity, sugar content, density, and alcohol percentage to determine its quality score.

***

## 📘 Project Overview

The goal of this project is to develop a predictive model that determines the quality of wine using measurable features.  
The model is trained on the **Wine Quality Dataset** and evaluated for accuracy on both training and testing data.

***

## 🧠 Features

- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA) using **Seaborn** and **Matplotlib**  
- Model training with **Logistic Regression**  
- Performance evaluation using accuracy metrics  
- Prediction based on user inputs  

***

## 📂 Dataset Information

The dataset includes various physicochemical properties of wine such as:  
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  
- Quality (target variable)

**Source:** [UCI Machine Learning Repository – Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

***

## ⚙️ Workflow

1. **Import Libraries**  
2. **Load Dataset**  
3. **Data Cleaning & Preprocessing**  
4. **Data Visualization**  
5. **Split Data into Train & Test Sets**  
6. **Model Training (Logistic Regression)**  
7. **Evaluate Model Accuracy**  
8. **Predict Based on User Input**

***

## 🚀 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/aarushdhawan11/Wine_Quality_Predictor.git

# 2. Navigate to the project folder
cd wine-quality-prediction

# 3. Install required libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# 4. Run the script
python wine_quality_prediction.py
```

When prompted, manually input the physicochemical values to get a wine quality prediction.

***

## 🧪 Sample Output

```
Enter fixed acidity: 7.4
Enter volatile acidity: 0.7
Enter citric acid: 0.0
Enter residual sugar: 1.9
Enter chlorides: 0.076
Enter free sulfur dioxide: 11
Enter total sulfur dioxide: 34
Enter density: 0.9978
Enter pH: 3.51
Enter sulphates: 0.56
Enter alcohol: 9.4

Predicted Result → Good Quality Wine 🍷
```

***

## 📊 Visualization Samples

This section displays key visualizations from the Wine Quality Prediction project along with corresponding Python code snippets.  
Save each plot as a `.png` file inside the `images/` folder and adjust paths as needed.

### 1. Wine Quality Count (Countplot)

**Code:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.savefig('images/catplot_quality_count.png')
```

**Visualization:**  
<img width="507" height="489" alt="0836b6d1-4090-4a72-bb99-3eb2aaedf9d7" src="https://github.com/user-attachments/assets/505cca39-d7cf-4ba4-82e9-893803d41b9d" />


### 2. Volatile Acidity vs Quality (Barplot)

**Code:**
```python
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.savefig('images/barplot_volatile_acidity.png')
```


**Visualization:**  
<img width="458" height="448" alt="e4dba20c-ce76-437f-bc69-dbce21620063" src="https://github.com/user-attachments/assets/1062ad6c-3d1c-4690-ae45-e2f4c78c9137" />


### 3. Citric Acid vs Quality (Barplot)

**Code:**
```python
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
plt.savefig('images/barplot_citric_acid.png')
```

**Visualization:**  
<img width="458" height="452" alt="f1e560a4-6320-4e73-988b-fac78fccf324" src="https://github.com/user-attachments/assets/2d8df7a7-9951-430d-9ba8-e05ecf5eafe7" />


### 4. Feature Correlation Heatmap

**Code:**
```python
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.savefig('images/correlation_heatmap.png')
```

**Visualization:**  
<img width="900" height="860" alt="83cc9b41-ca16-47c6-bd6f-5083f1710758" src="https://github.com/user-attachments/assets/bdafd213-eac6-464e-a165-48f8629a5703" />


## Model Accuracy

| Dataset | Accuracy |
|----------|-----------|
| Training | ~93% |
| Testing  | 93.4% |

***

## 🧰 Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

***

## 💡 Future Enhancements

- Deploy the model with **Flask** or **Streamlit**  
- Add a web-based user interface  
- Experiment with **XGBoost** or **Neural Networks**  
- Compare results using both red and white wine datasets  

***

## 📜 Author

**Aarush Dhawan**  
🎓 4th Year Student | 💻 Passionate about Data Science & Machine Learning  
📧 aarushdhawan25@gmail.com

***

## 🏷️ License

This project is open source under the **MIT License**.

***

