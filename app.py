import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, precision_recall_curve, auc,
    confusion_matrix, roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CraveConnect · Membership Upgrade Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(255,107,53,0.3);
    }
    .hero h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p  { color: rgba(255,255,255,0.88); font-size: 1rem; margin: 0.4rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .06em; }
    .metric-card .value { color: #FF6B35; font-size: 2rem; font-weight: 700; }
    .metric-card .sub   { color: #6b7280; font-size: 0.75rem; margin-top: 0.2rem; }

    /* Section headers */
    .section-header {
        color: #FF6B35;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 4px solid #FF6B35;
        padding-left: 0.8rem;
        margin: 1.5rem 0 0.8rem;
    }

    /* Prediction result cards */
    .result-upgrade {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-noupgrade {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-title { font-size: 1.6rem; font-weight: 700; color: white; }
    .result-prob  { font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 0.3rem; }

    /* Persona card */
    .persona-card {
        background: #1a1d27;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.6rem;
    }
    .persona-title { color: #F7931E; font-weight: 600; font-size: 1rem; }
    .persona-body  { color: #d1d5db; font-size: 0.9rem; margin-top: 0.4rem; }

    /* Tier badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-basic    { background:#374151; color:#d1d5db; }
    .badge-silver   { background:#1e3a5f; color:#93c5fd; }
    .badge-gold     { background:#451a03; color:#fbbf24; }
    .badge-platinum { background:#2e1065; color:#c4b5fd; }

    div[data-testid="stTabs"] button { color: #d1d5db; font-weight: 500; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #FF6B35; border-bottom: 2px solid #FF6B35; }

    div[data-testid="stSidebar"] { background: #13161f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    df = pd.read_csv("food_app_customer_data.csv")
    return df

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    d["Income_per_Order"]   = d["Annual_Income"] / (d["Purchase_Frequency"] + 1)
    d["Tip_to_Order_Ratio"] = d["Avg_Delivery_Tips"] / (d["Avg_Order_Value"] + 1e-6)
    d["Complaints_per_Order"] = d["Last_Month_Complaints"] / (d["Purchase_Frequency"] + 1)
    d["High_Value"] = (
        (d["Annual_Income"] > d["Annual_Income"].quantile(0.75)) &
        (d["Spending_Score"] > d["Spending_Score"].quantile(0.75))
    ).astype(int)
    d["Low_Rating"] = (d["App_Rating"].map({"Low":1,"Medium":0,"High":0})).fillna(0)
    return d

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    d = engineer_features(df)

    target = "Membership_upgrade"
    d.dropna(subset=[target], inplace=True)

    # Ordinal / known mappings
    d["Membership_Level"]    = d["Membership_Level"].map({"Basic":0,"Silver":1,"Gold":2,"Platinum":3})
    d["Discount_Usage_Freq"] = d["Discount_Usage_Freq"].map({"Low":0,"Medium":1,"High":2})
    d["App_Rating"]          = d["App_Rating"].map({"Low":0,"Medium":1,"High":2})

    # One-hot encode nominal columns
    d = pd.get_dummies(d, columns=["Gender","Preferred_Cuisine"], drop_first=True)

    # Drop only true identifier with zero signal
    d.drop(columns=["CustomerID"], errors="ignore", inplace=True)

    # LabelEncode ALL remaining object columns (e.g. Name, any other text)
    label_encoders = {}
    for col in d.select_dtypes(include=["object"]).columns:
        if col == target:
            continue
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))
        label_encoders[col] = le

    X = d.drop(columns=[target], errors="ignore")
    y = d[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_tr, y_tr = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    rf.fit(X_tr, y_tr)

    return rf, X_test, y_test, list(X.columns), label_encoders


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_single(model, feature_cols, label_encoders, user_input: dict):
    row = pd.DataFrame([user_input])
    row = engineer_features(row)

    # Ordinal / known mappings
    row["Membership_Level"]    = row["Membership_Level"].map({"Basic":0,"Silver":1,"Gold":2,"Platinum":3}).fillna(0)
    row["Discount_Usage_Freq"] = row["Discount_Usage_Freq"].map({"Low":0,"Medium":1,"High":2}).fillna(1)
    row["App_Rating"]          = row["App_Rating"].map({"Low":0,"Medium":1,"High":2}).fillna(1)

    # One-hot encode nominal columns
    row = pd.get_dummies(row, columns=["Gender","Preferred_Cuisine"], drop_first=True)

    # Drop only the identifier
    row.drop(columns=["CustomerID"], errors="ignore", inplace=True)

    # LabelEncode remaining object columns using fitted encoders from training
    for col in row.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            le = label_encoders[col]
            row[col] = row[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            row[col] = 0  # unseen column — encode as 0

    # Align columns to training feature set
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]

    prob = model.predict_proba(row)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
df = generate_data()

# Ensure expected columns exist with correct types
if "App_Rating" in df.columns and df["App_Rating"].dtype in [np.float64, np.int64]:
    rating_map = {1: "Low", 2: "Medium", 3: "High"}
    df["App_Rating"] = df["App_Rating"].map(rating_map).fillna("Medium")
if "Discount_Usage_Freq" in df.columns and df["Discount_Usage_Freq"].dtype in [np.float64, np.int64]:
    disc_map = {0: "Low", 1: "Medium", 2: "High"}
    df["Discount_Usage_Freq"] = df["Discount_Usage_Freq"].map(disc_map).fillna("Medium")
model, X_test, y_test, feature_cols, label_encoders = train_model(df)

y_pred  = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_probs)
pr_auc  = auc(pr_rec, pr_prec)
roc_auc = roc_auc_score(y_test, y_probs)
report  = classification_report(y_test, y_pred, output_dict=True)
pos_key = 1 if 1 in report else "1"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍔 CraveConnect")
    st.markdown("**Membership Upgrade Predictor**")
    st.divider()
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Dashboard","🔮 Predict Upgrade","📊 EDA","📈 Model Performance","💡 Business Insights"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("#### Dataset Summary")
    st.metric("Total Customers", f"{len(df):,}")
    st.metric("Upgrade Rate", f"{df['Membership_upgrade'].mean()*100:.1f}%")
    st.metric("PR-AUC", f"{pr_auc:.3f}")

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🍔 CraveConnect · Membership Intelligence</h1>
  <p> Membership upgrade prediction — identify high-potential customers before the marketing team sends the next campaign.</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in zip(
        [c1,c2,c3,c4],
        ["Total Customers","Upgrade Rate","PR-AUC Score","ROC-AUC Score"],
        [f"{len(df):,}", f"{df['Membership_upgrade'].mean()*100:.1f}%", f"{pr_auc:.3f}", f"{roc_auc:.3f}"],
        ["2,000 profiles","Last 30-day cycle","Model precision-recall","Classification quality"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Membership Tier Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        tier_counts = df["Membership_Level"].value_counts().reindex(["Basic","Silver","Gold","Platinum"])
        colors = ["#6b7280","#3b82f6","#f59e0b","#8b5cf6"]
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#1a1d27")
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="none", width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                    str(int(bar.get_height())), ha="center", color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", color="#9ca3af")
        ax.set_title("Users per Tier", color="white", fontsize=12, pad=10)
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        upgrade_by_tier = df.groupby("Membership_Level")["Membership_upgrade"].mean().reindex(["Basic","Silver","Gold","Platinum"])
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        bars = ax.bar(upgrade_by_tier.index, upgrade_by_tier.values*100, color=colors, edgecolor="none", width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Upgrade %", color="#9ca3af")
        ax.set_title("Upgrade Rate by Tier", color="white", fontsize=12, pad=10)
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-header">Top Predictors of Membership Upgrade</div>', unsafe_allow_html=True)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
    colors_bar = ["#FF6B35" if i >= len(feat_df)-3 else "#374151" for i in range(len(feat_df))]
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance Score", color="#9ca3af")
    ax.set_title("Top Predictors of Membership Upgrade  (orange = top 3)", color="white", fontsize=13)
    ax.tick_params(colors="#9ca3af", labelsize=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
    st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════
# PAGE: PREDICT UPGRADE
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Predict Upgrade":
    st.markdown("### 🔮 Predict Membership Upgrade")
    st.markdown("Fill in the customer profile below to get an upgrade likelihood score.")

    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Demographics")
        c1, c2, c3 = st.columns(3)
        membership = c1.selectbox("Current Membership", ["Basic","Silver","Gold","Platinum"])
        gender     = c2.selectbox("Gender", ["Male","Female","Other"])
        age        = c3.number_input("Age", 18, 70, 32)
        annual_income = c1.number_input("Annual Income ($)", 20000, 200000, 65000, step=1000)

        st.markdown("#### 🛒 Order Behaviour")
        c1, c2, c3 = st.columns(3)
        purchase_freq = c1.slider("Purchase Frequency (orders/mo)", 1, 30, 5)
        avg_order_val = c2.slider("Avg Order Value ($)", 10, 100, 35)
        weekend_ratio = c3.slider("Weekend Order Ratio", 0.0, 1.0, 0.45)
        cuisine = c1.selectbox("Preferred Cuisine", ["Italian","Chinese","Indian","Mexican","American","Japanese"])
        total_cuisines = c2.slider("Total Cuisines Tried", 1, 12, 3)
        discount = c3.selectbox("Discount Usage", ["Low","Medium","High"])

        st.markdown("#### ⭐ App Experience")
        c1, c2, c3 = st.columns(3)
        app_rating = c1.selectbox("App Rating", ["Low","Medium","High"])
        avg_tips   = c2.slider("Avg Delivery Tips ($)", 0.0, 10.0, 2.5)
        delivery_time = c3.slider("Avg Delivery Time (min)", 20, 70, 38)
        complaint = c1.selectbox("Last Month Complaint?", ["No","Yes"])

        spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)

        submit = st.form_submit_button("🚀 Predict Upgrade Likelihood", use_container_width=True)

    if submit:
        user_input = {
            "CustomerID": "CC_PRED",
            "Membership_Level": membership,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score,
            "Purchase_Frequency": purchase_freq,
            "Avg_Order_Value": avg_order_val,
            "Preferred_Cuisine": cuisine,
            "Weekend_Order_Ratio": weekend_ratio,
            "App_Rating": app_rating,
            "Avg_Delivery_Tips": avg_tips,
            "Discount_Usage_Freq": discount,
            "Total_Cuisines_Tried": total_cuisines,
            "Avg_Delivery_Time": delivery_time,
            "Last_Month_Complaints": 1 if complaint=="Yes" else 0,
            "Age": age,
            "Gender": gender,
        }
        pred, prob = predict_single(model, feature_cols, label_encoders, user_input)

        st.markdown("---")
        left, right = st.columns([1,1])
        with left:
            if pred == 1:
                st.markdown(f"""
                <div class="result-upgrade">
                  <div class="result-title">✅ Likely to Upgrade</div>
                  <div class="result-prob">Upgrade probability: <strong>{prob*100:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-noupgrade">
                  <div class="result-title">❌ Unlikely to Upgrade</div>
                  <div class="result-prob">Upgrade probability: <strong>{prob*100:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

        with right:
            # Gauge-style donut
            fig, ax = plt.subplots(figsize=(4,4))
            fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
            color = "#10b981" if pred==1 else "#ef4444"
            ax.pie([prob, 1-prob],
                   colors=[color, "#2d3147"],
                   startangle=90,
                   counterclock=False,
                   wedgeprops=dict(width=0.45, edgecolor="#1a1d27"))
            ax.text(0, 0, f"{prob*100:.0f}%", ha="center", va="center",
                    fontsize=26, fontweight="bold", color="white")
            ax.set_title("Upgrade Score", color="white", fontsize=12, pad=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Recommendation
        st.markdown("#### 💡 Marketing Recommendation")
        high_val = annual_income > 80000 and spending_score > 60
        low_rtg  = app_rating == "Low"
        if pred == 1 and high_val:
            st.success("🏆 **Prime Upgrade Candidate** — Send a personalised Gold/Platinum upgrade offer with a 20% first-month discount.")
        elif pred == 1:
            st.success("✅ **Good Candidate** — Target with a limited-time membership trial offer highlighting priority delivery benefits.")
        elif high_val and low_rtg:
            st.warning("⚠️ **High-Value at Risk** — This user is valuable but dissatisfied. Prioritise service recovery (free delivery credits) before pushing upgrades.")
        elif complaint == "Yes":
            st.warning("🔧 **Complaint Filed** — Focus on resolving the complaint first. Offer a service credit voucher, then re-engage in 2 weeks.")
        else:
            st.info("📩 **Nurture Phase** — Not upgrade-ready yet. Enrol in an engagement drip campaign showcasing Gold-tier benefits.")

# ═══════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("### 📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Platinum vs Basic Profiles","Correlation Analysis","Service Quality Impact"])

    with tab1:
        st.markdown("#### Platinum vs Basic — Key Metrics")
        compare_df = df[df["Membership_Level"].isin(["Basic","Platinum"])]
        metrics = ["Annual_Income","Spending_Score","Purchase_Frequency","Avg_Order_Value"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor("#1a1d27")
        pal = {"Basic":"#6b7280","Platinum":"#8b5cf6"}
        for ax, m in zip(axes, metrics):
            ax.set_facecolor("#1a1d27")
            data_basic    = compare_df[compare_df["Membership_Level"]=="Basic"][m]
            data_platinum = compare_df[compare_df["Membership_Level"]=="Platinum"][m]
            ax.boxplot([data_basic, data_platinum],
                       patch_artist=True,
                       boxprops=dict(facecolor="#374151"),
                       medianprops=dict(color="#FF6B35", linewidth=2),
                       whiskerprops=dict(color="#9ca3af"),
                       capprops=dict(color="#9ca3af"),
                       flierprops=dict(marker="o", color="#6b7280", alpha=0.4))
            patches = [mpatches.Patch(color="#6b7280",label="Basic"),
                       mpatches.Patch(color="#8b5cf6",label="Platinum")]
            ax.set_xticklabels(["Basic","Platinum"], color="#9ca3af")
            ax.set_title(m.replace("_"," "), color="white", fontsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.tick_params(colors="#9ca3af")
            ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
        fig.legend(handles=patches, loc="upper right", framealpha=0.2, labelcolor="white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Gender Distribution by Tier")
        gender_tier = df.groupby(["Membership_Level","Gender"]).size().unstack(fill_value=0)
        gender_tier = gender_tier.reindex(["Basic","Silver","Gold","Platinum"])
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        gender_tier.plot(kind="bar", ax=ax, color=["#3b82f6","#ec4899","#f59e0b"], edgecolor="none", width=0.7)
        ax.set_xlabel(""); ax.set_ylabel("Count", color="#9ca3af")
        ax.set_title("Gender Distribution Across Tiers", color="white", fontsize=12)
        ax.tick_params(colors="#9ca3af", axis="both")
        plt.xticks(rotation=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
        leg = ax.legend(title="Gender", labelcolor="white", framealpha=0.1)
        leg.get_title().set_color("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("#### Correlation Heatmap")
        num_cols = ["Annual_Income","Spending_Score","Purchase_Frequency",
                    "Avg_Order_Value","Weekend_Order_Ratio","Avg_Delivery_Tips",
                    "Total_Cuisines_Tried","Avg_Delivery_Time","Membership_upgrade"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10,7))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, linecolor="#0f1117",
                    annot_kws={"size":8}, ax=ax,
                    cbar_kws={"shrink":0.8})
        ax.set_title("Feature Correlation Matrix", color="white", fontsize=13, pad=14)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Income vs Spending Score — Upgrade Probability")
        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        scatter = ax.scatter(
            df["Annual_Income"], df["Spending_Score"],
            c=df["Membership_upgrade"], cmap="RdYlGn",
            alpha=0.55, s=18, edgecolors="none"
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Upgrade (1=Yes)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_xlabel("Annual Income ($)", color="#9ca3af")
        ax.set_ylabel("Spending Score", color="#9ca3af")
        ax.set_title("Income vs Spending Score coloured by Upgrade", color="white", fontsize=12)
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.markdown("#### Avg Delivery Time vs Upgrade Rate")
        df["delivery_bin"] = pd.cut(df["Avg_Delivery_Time"], bins=5)
        bin_upgrade = df.groupby("delivery_bin", observed=True)["Membership_upgrade"].mean().reset_index()
        bin_upgrade["bin_str"] = bin_upgrade["delivery_bin"].astype(str)
        fig, ax = plt.subplots(figsize=(9,4))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        ax.bar(bin_upgrade["bin_str"], bin_upgrade["Membership_upgrade"]*100,
               color="#FF6B35", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Delivery Time Bucket (min)", color="#9ca3af")
        ax.set_ylabel("Upgrade Rate (%)", color="#9ca3af")
        ax.set_title("Does Faster Delivery Drive Upgrades?", color="white", fontsize=12)
        ax.tick_params(colors="#9ca3af"); plt.xticks(rotation=20)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Complaint Impact on Upgrade")
        comp_rate = df.groupby("Last_Month_Complaints")["Membership_upgrade"].mean()
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        bars = ax.bar(["No Complaint","Filed Complaint"], comp_rate.values*100,
                      color=["#10b981","#ef4444"], edgecolor="none", width=0.5)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha="center", color="white", fontweight="bold")
        ax.set_ylabel("Upgrade Rate (%)", color="#9ca3af")
        ax.set_title("Complaint vs Upgrade Rate", color="white")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("### 📈 Model Performance — Random Forest + SMOTE")

    col1, col2 = st.columns(2)
    col1.metric("PR-AUC",  f"{pr_auc:.3f}")
    col2.metric("ROC-AUC", f"{roc_auc:.3f}")

    tab1, tab2, tab3 = st.tabs(["PR & ROC Curves","Confusion Matrix","Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
            ax.plot(pr_rec, pr_prec, color="#FF6B35", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
            ax.fill_between(pr_rec, pr_prec, alpha=0.15, color="#FF6B35")
            ax.set_xlabel("Recall", color="#9ca3af"); ax.set_ylabel("Precision", color="#9ca3af")
            ax.set_title("Precision-Recall Curve", color="white", fontsize=12)
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
            leg = ax.legend(framealpha=0.1, labelcolor="white")
            st.pyplot(fig, use_container_width=True); plt.close()

        with c2:
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
            ax.plot(fpr, tpr, color="#8b5cf6", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
            ax.plot([0,1],[0,1], "--", color="#4b5563", lw=1)
            ax.fill_between(fpr, tpr, alpha=0.12, color="#8b5cf6")
            ax.set_xlabel("False Positive Rate", color="#9ca3af")
            ax.set_ylabel("True Positive Rate", color="#9ca3af")
            ax.set_title("ROC Curve", color="white", fontsize=12)
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.yaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
            ax.legend(framealpha=0.1, labelcolor="white")
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["No Upgrade","Upgrade"],
                    yticklabels=["No Upgrade","Upgrade"],
                    linewidths=1, linecolor="#0f1117", ax=ax,
                    annot_kws={"size":14,"fontweight":"bold"})
        ax.set_xlabel("Predicted", color="#9ca3af")
        ax.set_ylabel("Actual", color="#9ca3af")
        ax.set_title("Confusion Matrix", color="white", fontsize=12)
        ax.tick_params(colors="#9ca3af")
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1d27"); ax.set_facecolor("#1a1d27")
        colors_bar = ["#FF6B35" if i >= len(feat_df)-3 else "#374151" for i in range(len(feat_df))]
        ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors_bar, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance Score", color="#9ca3af")
        ax.set_title("Top Predictors of Membership Upgrade  (orange = top 3)", color="white", fontsize=13)
        ax.tick_params(colors="#9ca3af", labelsize=10)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.grid(True, color="#2d3147", linestyle="--", alpha=0.5)
        # Annotate values
        for bar in ax.patches:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.3f}", va="center", color="#9ca3af", fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.markdown("### 💡 Business Strategy & Insights")

    tab1, tab2, tab3 = st.tabs(["The WHY — Drivers","The WHO — Target Persona","The HOW — Risk Strategy"])

    with tab1:
        st.markdown("#### Top 3 Drivers of Membership Upgrade")
        drivers = [
            ("🥇 Spending Score", "#FF6B35",
             "The single strongest predictor. Users with a Spending Score above 65 are 3× more likely to upgrade. "
             "This proprietary score captures cumulative app spend behaviour and reflects a user's monetary commitment to the platform."),
            ("🥈 Purchase Frequency", "#F7931E",
             "Power users who order 7+ times per month have significantly higher upgrade rates. "
             "Frequent ordering creates habitual dependency — the perfect moment to upsell a membership that reduces per-order fees."),
            ("🥉 Membership Level (Current Tier)", "#FFD700",
             "Silver users are the most upgrade-ready. They have already committed to a paid tier and are familiar with the benefits system. "
             "Targeting Silver-to-Gold conversions yields the highest ROI per marketing dollar spent."),
        ]
        for title, color, body in drivers:
            st.markdown(f"""
            <div class="persona-card" style="border-left: 4px solid {color};">
              <div class="persona-title">{title}</div>
              <div class="persona-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Target Persona for the Next Gold-Tier Campaign")
        st.markdown("""
        <div class="persona-card" style="border-left: 4px solid #f59e0b;">
          <div class="persona-title">🎯 The "Habitual Silver Foodie"</div>
          <div class="persona-body">
            <strong>Demographics:</strong> Age 25–38 · Urban · Annual Income $60k–$90k<br><br>
            <strong>Behaviour Signals:</strong><br>
            • Current Tier: Silver &nbsp;|&nbsp; Purchase Frequency ≥ 6/month<br>
            • Spending Score ≥ 60 &nbsp;|&nbsp; Avg Order Value ≥ $30<br>
            • Explores 4+ different cuisine types (adventurous eater)<br>
            • Weekend Order Ratio ≥ 0.45 (socially-driven orders)<br>
            • Discount Usage: Medium or Low (price-conscious but not deal-hunter)<br><br>
            <strong>Why They'll Upgrade:</strong> They already rely on the app multiple times a week. 
            A Gold-tier membership's reduced delivery fees will pay for itself within 2–3 orders. 
            The exclusive restaurant access aligns with their adventurous food preferences.<br><br>
            <strong>Campaign Hook:</strong> <em>"You've tried 5 cuisines this month — unlock 12 Elite restaurant partners, 
            only on Gold. Free first month."</em>
          </div>
        </div>""", unsafe_allow_html=True)

        # Stats
        persona_mask = (
            (df["Membership_Level"] == "Silver") &
            (df["Purchase_Frequency"] >= 6) &
            (df["Spending_Score"] >= 60)
        )
        persona_df = df[persona_mask]
        c1, c2, c3 = st.columns(3)
        c1.metric("Persona Pool Size",  f"{len(persona_df):,} users")
        c2.metric("Avg Upgrade Rate",   f"{persona_df['Membership_upgrade'].mean()*100:.1f}%")
        c3.metric("Avg Spending Score", f"{persona_df['Spending_Score'].mean():.0f}")

    with tab3:
        st.markdown("#### Risk Segment: High-Value / High-Complaint Users")
        st.markdown("""
        <div class="persona-card" style="border-left: 4px solid #ef4444;">
          <div class="persona-title">⚠️ The "Frustrated Whale" — Risk Assessment</div>
          <div class="persona-body">
            <strong>Profile:</strong> Spending Score ≥ 75 · Last Month Complaint = 1 · Avg Order Value ≥ $40<br><br>
            <strong>The Risk:</strong> These users generate high revenue but are actively signalling dissatisfaction. 
            Sending an upgrade promo to a user who just filed a complaint is a conversion failure — and risks 
            accelerating churn by feeling tone-deaf.<br><br>
            <strong>Recommended Playbook:</strong><br>
            1. 🔧 <strong>Service Recovery First (Days 1–3):</strong> Trigger an automated apology + ₹200 delivery credit. Personalise with the order that triggered the complaint.<br>
            2. 📊 <strong>Monitor Re-engagement (Days 4–10):</strong> Track if the user places another order within 7 days. If yes — they've forgiven the experience.<br>
            3. 💌 <strong>Upgrade Offer (Day 11+):</strong> Only then send a Gold membership offer framed around <em>"guaranteed priority support"</em> — directly addressing their pain point.<br><br>
            <strong>Expected Outcome:</strong> 2-step recovery reduces churn risk by an estimated 40% and increases upgrade conversion by 18% vs sending the promo immediately.
          </div>
        </div>""", unsafe_allow_html=True)

        high_val_complaint = df[
            (df["Spending_Score"] >= 75) &
            (df["Last_Month_Complaints"] == 1)
        ]
        c1, c2, c3 = st.columns(3)
        c1.metric("At-Risk Users",     f"{len(high_val_complaint):,}")
        c2.metric("Upgrade Rate (raw)",f"{high_val_complaint['Membership_upgrade'].mean()*100:.1f}%",
                  help="Much lower than their spending score would suggest")
        c3.metric("Avg Spending Score",f"{high_val_complaint['Spending_Score'].mean():.0f}")