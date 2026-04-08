import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from prophet import Prophet

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Inventory System", layout="wide")

# ----------------------------
# DB CONNECTION
# ----------------------------
try:
    engine = create_engine(st.secrets["DB_URL"])
except:
    st.error("❌ Database connection failed.")
    st.stop()

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM inventory", engine)

df = load_data()

if df.empty:
    st.warning("⚠️ No data found.")
    st.stop()

# ----------------------------
# DATA VARIABILITY
# ----------------------------
np.random.seed(42)

category_factor = {
    "vegetables": 1.2,
    "fruits": 0.9,
    "dairy": 1.5
}

df['category_factor'] = df['category'].map(category_factor).fillna(1)

df['demand'] = np.random.lognormal(
    mean=np.log(350 * df['category_factor']),
    sigma=0.35,
    size=len(df)
).clip(50, 800)

df['demand'] = df['demand'] * np.random.uniform(0.7, 1.4, len(df))

df['available_quantity'] = np.random.randint(100, 1200, len(df))
df['lead_time'] = np.random.randint(2, 10, len(df))
df['demand_std'] = df['demand'] * np.random.uniform(0.15, 0.4, len(df))

# ----------------------------
# INVENTORY CALCULATIONS
# ----------------------------
S = 50
Z = 1.65

df['holding_cost'] = (0.15 * df['mrp']).clip(5, 50)

df['EOQ'] = np.sqrt((2 * df['demand'] * S) / df['holding_cost'])

df['safety_stock'] = Z * df['demand_std'] * np.sqrt(df['lead_time'])
df['ROP'] = (df['demand'] * df['lead_time']) + df['safety_stock']

df['reorder_flag'] = df['available_quantity'] < df['ROP']

df['recommendation'] = df.apply(
    lambda x: f"Order {int(x['EOQ'])}" if x['reorder_flag'] else "OK",
    axis=1
)

# ----------------------------
# DEMAND CATEGORY
# ----------------------------
df['demand_category'] = pd.cut(
    df['demand'],
    bins=[0, 200, 400, 600, 800, 1000],
    labels=["Low", "Moderate", "High", "Very High", "Extreme"]
)

# ----------------------------
# FORECAST FUNCTION (REALISTIC)
# ----------------------------
def generate_timeseries(product_name, base_demand):
    periods = 60
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods)

    seed = abs(hash(product_name + str(base_demand))) % (10**6)
    np.random.seed(seed)

    trend = np.linspace(0, np.random.randint(-50, 100), periods)

    seasonality = 30 * np.sin(np.linspace(0, 6*np.pi, periods))
    seasonality += np.random.normal(0, 10, periods)

    noise = np.random.normal(0, base_demand * 0.1, periods)

    y = base_demand + trend + seasonality + noise
    y = np.clip(y, base_demand * 0.5, base_demand * 1.8)

    return pd.DataFrame({'ds': dates, 'y': y})


def forecast_demand(product_name, filtered_df):
    product_data = filtered_df[filtered_df['product_name'] == product_name]

    if product_data.empty:
        return None

    base_demand = product_data['demand'].values[0]

    ts = generate_timeseries(product_name, base_demand)

    model = Prophet()
    model.fit(ts)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    return forecast

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("📦 Inventory System")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

menu = st.sidebar.radio("Navigation", ["Dashboard", "Insights", "Data"])

category = st.sidebar.selectbox("Category", df['category'].dropna().unique())
filtered_df = df[df['category'] == category]

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.title("📊 Inventory Dashboard")

    # KPI
    col1, col2, col3 = st.columns(3)

    col1.metric("📦 Total Products", len(df))
    col2.metric("⚠️ Reorders Needed", int(df['reorder_flag'].sum()))
    col3.metric("🚨 Critical Items", len(df[df['available_quantity'] < df['ROP']*0.8]))

    st.divider()

    col1, col2 = st.columns(2)

    # 🔥 TOP PRODUCTS (FIXED UI)
    with col1:
        st.subheader("Top Demand Products")

        top_products = filtered_df.sort_values(by='demand', ascending=False).head(10)

        top_products['status'] = np.where(
            top_products['available_quantity'] < top_products['ROP'],
            "Critical",
            "Normal"
        )

        fig = px.bar(
            top_products,
            y='product_name',
            x='demand',
            color='status',
            orientation='h',
            color_discrete_map={
                "Normal": "#4C78A8",
                "Critical": "#E45756"
            },
            template='plotly_dark'
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))

        st.plotly_chart(fig, use_container_width=True)

    # STOCK VS ROP
    with col2:
        st.subheader("Stock vs ROP")

        compare = filtered_df[['product_name', 'available_quantity', 'ROP']].head(10)

        fig2 = px.bar(
            compare,
            y='product_name',
            x=['available_quantity', 'ROP'],
            orientation='h',
            barmode='group',
            template='plotly_dark'
        )

        fig2.update_layout(yaxis=dict(autorange="reversed"))

        st.plotly_chart(fig2, use_container_width=True)

    # FORECAST
    st.subheader("📈 Demand Forecast")

    selected_product = st.selectbox("Select Product", filtered_df['product_name'].unique())

    forecast = forecast_demand(selected_product, filtered_df)

    if forecast is not None:
        fig_forecast = px.line(forecast, x='ds', y='yhat', template='plotly_dark')
        st.plotly_chart(fig_forecast, use_container_width=True)

    # DISTRIBUTION
    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Category Demand Share")

        cat_data = df.groupby('category')['demand'].sum().reset_index()

        fig_pie = px.pie(cat_data, names='category', values='demand', template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        st.subheader("Demand Distribution")

        fig_hist = px.histogram(
            df,
            x='demand_category',
            color='demand_category',
            template='plotly_dark'
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # HEATMAP (FIXED)
    st.subheader("Category Performance Heatmap")

    heatmap_data = df.groupby('category')[['demand', 'available_quantity', 'ROP']].mean()

    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    fig_heatmap = px.imshow(
        heatmap_norm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdYlBu'
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

# ----------------------------
# INSIGHTS
# ----------------------------
elif menu == "Insights":

    st.title("📊 Inventory Insights")

    alerts = filtered_df[filtered_df['reorder_flag']]

    if len(alerts) > 0:
        st.error(f"⚠️ {len(alerts)} products need reorder!")
    else:
        st.success("Inventory is healthy ✅")

    st.dataframe(alerts[['product_name', 'available_quantity', 'ROP', 'EOQ']])

# ----------------------------
# DATA
# ----------------------------
elif menu == "Data":

    st.title("📋 Inventory Data")

    st.dataframe(filtered_df)
