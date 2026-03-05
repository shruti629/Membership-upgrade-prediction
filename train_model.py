import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# Create outputs folder
# =========================

os.makedirs("outputs", exist_ok=True)


# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/food_app_customer_data.csv")

print("Dataset Loaded Successfully")


# =========================
# Encode Categorical Variables
# =========================

# One-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Preferred_Cuisine'], drop_first=True)

# Label encoding
membership_map = {'Basic':0, 'Silver':1, 'Gold':2, 'Platinum':3}
discount_map = {'Low':0, 'Medium':1, 'High':2}

df['Membership_Level'] = df['Membership_Level'].map(membership_map)
df['Discount_Usage_Freq'] = df['Discount_Usage_Freq'].map(discount_map)

print("Categorical variables encoded successfully")


# =========================
# Preprocessing
# =========================

df_processed = df.copy()

# Drop rows where target missing
df_processed.dropna(subset=['Membership_upgrade'], inplace=True)


drop_features = [
    'Avg_Delivery_Tips',
    'Last_Month_Complaints',
    'Annual_Income',
    'Age',
    'App_Rating',
    'Total_Cuisines_Tried',
    'CustomerID',
    'Name',
    'Gender_num'
]

target = "Membership_upgrade"

y = df_processed[target]

X = df_processed.drop(columns=drop_features + [target], errors="ignore")

print("\nFeatures used for modeling:")
print(X.columns)


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Handle Class Imbalance
# =========================

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nTraining set size after SMOTE:", X_train_res.shape)


# =========================
# Create Model
# =========================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


# =========================
# Train Model
# =========================

model.fit(X_train_res, y_train_res)

print("\nModel Training Completed")


# =========================
# Save Model
# =========================

pickle.dump(model, open("outputs/model.pkl", "wb"))
pickle.dump(X.columns, open("outputs/features.pkl", "wb"))

print("\nModel saved successfully in outputs/")


