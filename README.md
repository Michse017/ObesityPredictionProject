# 🧠 Project: Obesity Prediction with Machine Learning

This project aims to train a machine learning model that, based on personal and lifestyle data, can predict a person's level of obesity.

The model is trained but **not used to make individual predictions** at this stage. The focus is on the **training and evaluation process** of the model.

---

## 📁 Project Structure

```
ObesityPredictionProject/
│
├── data/
│   └── ObesityDataSet.csv         # Original dataset with 2111 records
│
├── src/
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   └── model.py                   # Model training and evaluation
│
├── main.py                        # Main script that runs the entire process
└── README.md                      # Project documentation (this file)
```
---

## ⚙️ What does the code do?

1. **Data loading:** Loads 2111 records from a CSV file.
2. **Preprocessing:**
   - Converts categorical variables into numerical format.
   - Imputes missing values using the mean.
   - Any remaining NaNs are filled with zeros.
3. **Data splitting:**
   - 70% for training (1477 records)
   - 25% for testing (528 records)
   - 5% for final evaluation (106 records)
4. **Model training:** Trains a linear regression model using the training set.
5. **Model evaluation:**
   - Evaluates the model on the test and final evaluation sets.
   - Reports the **Mean Squared Error (MSE)** as the evaluation metric.

---

## 📊 Sample Results

Original dataset row count: 2111  
Some NaN values remained after imputing with the mean. They were filled with 0.  
Row count after preprocessing: 2111  
Training set size: 1477  
Test set size: 528  
Final evaluation set size: 106  
Model trained successfully.  

Test set evaluation:  
Model evaluation:  
Mean Squared Error (MSE): 0.2023  

Final evaluation set:  
Model evaluation:  
Mean Squared Error (MSE): 0.1976  

---

## 🛠️ Libraries Used

- pandas  
- numpy  
- scikit-learn  

---

## ✅ Project Status

✔️ Model training and evaluation working  
❌ Does not make individual predictions yet  

---

## 📌 Notes

This project was developed as part of a university workshop on Artificial Intelligence. The main objective is to demonstrate the process of training and evaluating a predictive model using real-world data.
