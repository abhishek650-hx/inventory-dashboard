import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Inventory System", layout="wide")

# ----------------------------
# DB CONNECTION
# ----------------------------
try:
    engine = create_engine(st.secrets["DB_URL"])
except Exception as e:
    st.error("❌ Database connection failed. Check DB_URL in secrets.")
    st.stop()

# ----------------------------
# CUSTOM UI STYLING
# ----------------------------
st.markdown("""
<style>
.stApp {background-color: #0e1117; color: white;}
section[data-testid="stSidebar"] {background-color: #111827;}
[data-testid="metric-container"] {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #374151;
}
h1, h2, h3 {color: #f9fafb;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.subheader("📈 Demand Forecast")

selected_product = st.selectbox(
    "Select Product for Forecast",
    df['product_name'].unique()
)

forecast = forecast_demand(df, selected_product)

fig_forecast = px.line(
    forecast,
    x='ds',
    y='yhat',
    title="7-Day Demand Forecast",
    template='plotly_dark'
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ----------------------------
# LOAD DATA FROM DATABASE
# ----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM inventory", engine)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# ----------------------------
# HANDLE EMPTY DATA
# ----------------------------
if df.empty:
    st.warning("⚠️ No data found in database.")
    st.info("👉 Check if data is uploaded in Supabase OR DB_URL is correct.")
    st.stop()

# ----------------------------
# CALCULATIONS
# ----------------------------
df['demand'] = df['demand'].replace(0, 1)

S = 50
df['holding_cost'] = 0.1 * df['mrp']
df['holding_cost'] = df['holding_cost'].replace(0, 0.1)

df['EOQ'] = np.sqrt((2 * df['demand'] * S) / df['holding_cost'])
df['EOQ'] = df['EOQ'].replace([np.inf, -np.inf], np.nan).fillna(0)

lead_time = 2
df['safety_stock'] = df['demand'].std() * np.sqrt(lead_time)

df['ROP'] = (df['demand'].mean() * lead_time) + df['safety_stock']

df['reorder_flag'] = df['available_quantity'] < df['ROP']

df['recommendation'] = df.apply(
    lambda x: f"Order {max(1, int(x['EOQ']))}" if x['reorder_flag'] else "OK",
    axis=1
)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 📦 Inventory System")
st.sidebar.markdown("---")

# 🔄 Refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# 📂 CSV Upload (Optional update)
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        new_df = pd.read_csv(uploaded_file)

        new_df.rename(columns={
            'name': 'product_name',
            'Category': 'category',
            'quantity': 'demand',
            'availableQuantity': 'available_quantity'
        }, inplace=True)

        new_df = new_df[['product_name', 'category', 'mrp', 'demand', 'available_quantity']]
        new_df['last_updated'] = pd.Timestamp.now()

        new_df.to_sql("inventory", engine, if_exists="replace", index=False)

        st.sidebar.success("✅ Database updated successfully!")

        st.cache_data.clear()
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"❌ Upload failed: {e}")

# Navigation
menu = st.sidebar.radio("Navigation", ["Dashboard", "Insights", "Data"])

category = st.sidebar.selectbox("Category", df['category'].dropna().unique())
filtered_df = df[df['category'] == category]

from prophet import Prophet

def forecast_demand(df, product_name):
    product_df = df[df['product_name'] == product_name].copy()

    # Create fake date column (since we don't have time series yet)
    product_df['ds'] = pd.date_range(end=pd.Timestamp.today(), periods=len(product_df))
    product_df['y'] = product_df['demand']

    model = Prophet()
    model.fit(product_df[['ds', 'y']])

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    return forecast

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.title("📊 Inventory Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Products", len(df))
    col2.metric("⚠️ Reorders", int(df['reorder_flag'].sum()))
    col3.metric("📊 Avg Demand", round(df['demand'].mean(), 2))
    col4.metric("📉 Avg EOQ", round(df['EOQ'].mean(), 2))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Demand Products")
        top_products = filtered_df.sort_values(by='demand', ascending=False).head(10)

        fig = px.bar(
            top_products,
            x='product_name',
            y='demand',
            color='demand',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stock vs ROP")

        compare = filtered_df[['product_name', 'available_quantity', 'ROP']].head(10)

        fig2 = px.line(
            compare,
            x='product_name',
            y=['available_quantity', 'ROP'],
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Category Demand")
        cat = df.groupby('category')['demand'].sum().reset_index()

        fig3 = px.pie(cat, names='category', values='demand', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Stock Distribution")
        fig4 = px.histogram(filtered_df, x='available_quantity', nbins=20, template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# INSIGHTS
# ----------------------------
elif menu == "Insights":

    st.title("📊 Inventory Insights")

    alerts = filtered_df[filtered_df['reorder_flag'] == True]

    if len(alerts) > 0:
        st.error(f"⚠️ {len(alerts)} products need reorder!")
    else:
        st.success("Inventory is healthy ✅")

    st.subheader("Critical Products")
    st.dataframe(alerts[['product_name', 'available_quantity', 'ROP', 'EOQ']], use_container_width=True)

    st.subheader("Recommendations")
    st.dataframe(filtered_df[['product_name', 'recommendation']], use_container_width=True)

# ----------------------------
# DATA
# ----------------------------
elif menu == "Data":

    st.title("📋 Inventory Data")

    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        "📥 Download Data",
        csv,
        "inventory.csv",
        "text/csv"
    )
