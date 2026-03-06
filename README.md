#  📊 Project Overview

Membership upgrades are important for businesses because they increase  **customer lifetime value and revenue.**

This project builds a **predictive system** that analyzes customer behavior and predicts whether a user is likely to upgrade their membership.

The application provides:

* Customer upgrade prediction

* Model insights

* Feature importance analysis

* Interactive dashboard

The web interface is built using **Streamlit**

# 🎯 Objectives

* Predict membership upgrade probability

* Identify key drivers influencing upgrades

* Help businesses design targeted marketing campaigns

* Provide an interactive ML dashboard

# ⚙️ Tech Stack

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Core programming language |
| Pandas & NumPy          | Data processing           |
| Scikit-learn            | Machine learning models   |
| XGBoost / Random Forest | Predictive modeling       |
| SHAP                    | Model explainability      |
| Matplotlib / Seaborn    | Data visualization        |
| Streamlit               | Web app deployment        |
| GitHub                  | Version control           |


# 🧠 Machine Learning Workflow

The project follows a complete **ML pipeline**:

## 1️⃣ Data Collection

Customer dataset including behavioral and demographic features such as:

* Gender

* Spending Score

* Purchase Frequency

* Preferred Cuisine

* Weekend Order Ratio

* Delivery Tips

* Complaints

* Delivery Time

* Income per Order

## 2️⃣ Data Preprocessing

* Handling missing values

* Feature engineering

* Encoding categorical variables

* Feature scaling

* Train-test split

## 3️⃣ Model Training

Different machine learning models were tested:

* Random Forest

* XGBoost (Best performance)

## 4️⃣ Model Evaluation

Evaluation metrics used:

* Accuracy

* Precision

* Recall

* F1-score

* ROC-AUC

* Precision-Recall AUC

## 5️⃣ Explainability

The project uses **SHAP values** to understand:

* Which features influence upgrade decisions

* Customer behavior patterns

* Key drivers of membership upgrades

# 📈 Key Insights

Some important insights discovered from the model:

* Customers with **higher spending scores** are more likely to upgrade.

* Frequent orders increase upgrade probability.

* Customers with fewer complaints show higher upgrade likelihood.

* Delivery satisfaction influences membership decisions.

# 🖥️ Web Application Features

The Streamlit application includes:

✔ Interactive prediction interface

✔ Input customer details

✔ Real-time upgrade prediction

✔ Feature importance 

✔ Business insights dashboard

# 🌐 Live Demo

Try the app here:

👉 https://membership-upgrade-prediction-u8jushslzput2dfcfkl4cr.streamlit.app/


# 🛠 Installation

Clone the repository:

    git clone https://github.com/shruti629/Membership-upgrade-prediction.git
    
    cd Membership-upgrade-prediction


Install dependencies:

    pip install -r requirements.txt


Run the Streamlit app:

    streamlit run app.py

# 📊 Use Cases

This system can be used by:

* Subscription businesses

* Food delivery platforms

* E-commerce membership programs

* SaaS platforms

* Customer retention teams

# 🚀 Future Improvements

* Real-time customer data integration

* Customer segmentation using clustering

* Personalized promotion recommendations


⭐ If you like this project, please **star the repository**!
