import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from prophet import Prophet

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Inventory Intelligence Dashboard", layout="wide")

# ----------------------------
# DB CONNECTION
# ----------------------------
try:
    engine = create_engine(st.secrets["DB_URL"])
except:
    st.error("Database connection failed")
    st.stop()

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM inventory", engine)

df = load_data()

if df.empty:
    st.warning("No data available")
    st.stop()

# ----------------------------
# DATA ENGINEERING
# ----------------------------
np.random.seed(42)

df['demand'] = np.random.lognormal(mean=np.log(350), sigma=0.35, size=len(df)).clip(50, 800)
df['demand'] *= np.random.uniform(0.7, 1.4, len(df))

df['available_quantity'] = np.random.randint(100, 1200, len(df))
df['lead_time'] = np.random.randint(2, 10, len(df)
)
df['demand_std'] = df['demand'] * np.random.uniform(0.15, 0.4, len(df))

# ----------------------------
# INVENTORY LOGIC
# ----------------------------
S = 50
Z = 1.65

df['holding_cost'] = (0.15 * df['mrp']).clip(5, 50)

df['EOQ'] = np.sqrt((2 * df['demand'] * S) / df['holding_cost'])
df['safety_stock'] = Z * df['demand_std'] * np.sqrt(df['lead_time'])
df['ROP'] = (df['demand'] * df['lead_time']) + df['safety_stock']

df['reorder_flag'] = df['available_quantity'] < df['ROP']

# Demand categories
df['demand_category'] = pd.cut(
    df['demand'],
    bins=[0, 200, 400, 600, 800, 1000],
    labels=["Low", "Moderate", "High", "Very High", "Extreme"]
)

# ----------------------------
# FORECAST FUNCTION
# ----------------------------
def generate_timeseries(product_name, base_demand):
    periods = 60
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods)

    np.random.seed(abs(hash(product_name)) % (10**6))

    trend = np.linspace(0, np.random.randint(-50, 100), periods)
    seasonality = 30 * np.sin(np.linspace(0, 6*np.pi, periods))
    seasonality += np.random.normal(0, 10, periods)

    noise = np.random.normal(0, base_demand * 0.1, periods)

    y = base_demand + trend + seasonality + noise
    y = np.clip(y, base_demand * 0.5, base_demand * 1.8)

    return pd.DataFrame({'ds': dates, 'y': y})

def forecast_demand(product_name):
    base = df[df['product_name'] == product_name]['demand'].values[0]
    ts = generate_timeseries(product_name, base)

    model = Prophet()
    model.fit(ts)

    future = model.make_future_dataframe(periods=7)
    return model.predict(future)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Inventory Controls")

category = st.sidebar.selectbox("Select Category", df['category'].unique())
filtered_df = df[df['category'] == category]

menu = st.sidebar.radio("View", ["Dashboard", "Insights", "Data"])

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.title("Inventory Intelligence Dashboard")

    # KPI CARDS
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", len(df))
    col2.metric("Reorders Needed", int(df['reorder_flag'].sum()))
    col3.metric("Critical Items", len(df[df['available_quantity'] < df['ROP']*0.8]))

    st.divider()

    col1, col2 = st.columns(2)

    # TOP PRODUCTS
    with col1:
        st.subheader("Top Demand Products")

        top = filtered_df.sort_values(by='demand', ascending=False).head(10)

        top['status'] = np.where(
            top['available_quantity'] < top['ROP'],
            "Critical", "Normal"
        )

        fig = px.bar(
            top,
            y='product_name',
            x='demand',
            color='status',
            orientation='h',
            color_discrete_map={"Normal": "#4C78A8", "Critical": "#E45756"},
            labels={"demand": "Demand (Units)"}
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # STOCK VS ROP
    with col2:
        st.subheader("Stock vs Reorder Point")

        fig2 = px.bar(
            filtered_df.head(10),
            y='product_name',
            x=['available_quantity', 'ROP'],
            orientation='h',
            barmode='group',
            labels={
                "value": "Stock Level (Units)",
                "variable": "Metric"
            }
        )

        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    # FORECAST
    st.subheader("Demand Forecast")

    product = st.selectbox("Select Product", filtered_df['product_name'].unique())
    forecast = forecast_demand(product)

    fig3 = px.line(forecast, x='ds', y='yhat', labels={"yhat": "Forecasted Demand"})
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    # PIE
    with col3:
        st.subheader("Category Share")

        fig4 = px.pie(
            df.groupby('category')['demand'].sum().reset_index(),
            names='category',
            values='demand'
        )
        st.plotly_chart(fig4, use_container_width=True)

    # HISTOGRAM
    with col4:
        st.subheader("Demand Distribution")

        fig5 = px.histogram(
            df,
            x='demand_category',
            color='demand_category'
        )
        st.plotly_chart(fig5, use_container_width=True)

    # HEATMAP (FIXED)
    st.subheader("Category Performance")

    heat = df.groupby('category')[['demand', 'available_quantity', 'ROP']].mean()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min())

    fig6 = px.imshow(
        heat_norm,
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig6, use_container_width=True)

# ----------------------------
# INSIGHTS
# ----------------------------
elif menu == "Insights":
    st.title("Inventory Alerts")

    alerts = filtered_df[filtered_df['reorder_flag']]
    st.dataframe(alerts)

# ----------------------------
# DATA
# ----------------------------
elif menu == "Data":
    st.title("Raw Data")
    st.dataframe(filtered_df)
