import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Analytics Dashboard")

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
high_risk_customers = pd.read_csv("high_risk_customers.csv")
model = joblib.load("churn_model.pkl")
feature_importance = pd.read_csv("feature_importance.csv")
feature_names = model.get_booster().feature_names
# Sidebar filters
st.sidebar.header("Filters")

contract_filter = st.sidebar.multiselect(
    "Select Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "Select Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

# Apply filters
filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter))
]

# KPI Metrics
total_customers = filtered_df.shape[0]
churn_rate = filtered_df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
avg_monthly = filtered_df["MonthlyCharges"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

st.divider()
# Top churn drivers
st.subheader("Top Churn Drivers")

top_features = feature_importance.head(10)

fig4 = px.bar(
    top_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Features Influencing Customer Churn"
)

fig4.update_layout(yaxis={'categoryorder':'total ascending'})

st.plotly_chart(fig4, use_container_width=True)
# Charts section
col4, col5 = st.columns(2)

with col4:
    fig1 = px.histogram(
        filtered_df,
        x="Contract",
        color="Churn",
        title="Churn by Contract Type",
        barmode="group"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    fig2 = px.histogram(
        filtered_df,
        x="InternetService",
        color="Churn",
        title="Churn by Internet Service",
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Monthly Charges vs Churn
fig3 = px.box(
    filtered_df,
    x="Churn",
    y="MonthlyCharges",
    title="Monthly Charges vs Churn"
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# High risk customers
st.subheader("⚠ High Risk Customers")

st.dataframe(
    high_risk_customers,
    use_container_width=True
)

st.caption("Customers with churn probability > 70%")
st.divider()

# ==============================
# 🔮 Churn Prediction Tool
# ==============================

st.subheader(" Predict Customer Churn")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 18, 120, 70)
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

# Convert input into dataframe
# Create empty dataframe with all training features
input_data = pd.DataFrame(columns=feature_names)

# Fill with zeros
input_data.loc[0] = 0

# Add user inputs
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges

# Prediction button
if st.button("Predict Churn Risk"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ High Churn Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Churn Risk ({probability*100:.2f}%)")