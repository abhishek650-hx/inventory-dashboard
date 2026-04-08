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
# 🔥 FIXED DATA VARIABILITY
# ----------------------------
np.random.seed(42)

category_factor = {
    "vegetables": 1.2,
    "fruits": 0.9,
    "dairy": 1.5
}

df['category_factor'] = df['category'].map(category_factor).fillna(1)

# ✅ FIXED DISTRIBUTION (LESS SKEWED)
df['demand'] = np.random.lognormal(
    mean=np.log(350 * df['category_factor']),
    sigma=0.35,
    size=len(df)
).clip(50, 800)

# ✅ ADD PRODUCT-LEVEL VARIABILITY
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
# FORECAST FUNCTION
# ----------------------------
def generate_timeseries(product_name, base_demand):
    periods = 60
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods)

    seed = abs(hash(product_name + str(base_demand))) % (10**6)
    np.random.seed(seed)

    trend_type = np.random.choice(["up", "down", "flat"])

    if trend_type == "up":
        trend = np.linspace(0, np.random.randint(20, 120), periods)
    elif trend_type == "down":
        trend = np.linspace(0, -np.random.randint(20, 120), periods)
    else:
        trend = np.zeros(periods)

    season_type = np.random.choice(["weekly", "irregular", "none"])

    if season_type == "weekly":
        seasonality = 40 * np.sin(np.linspace(0, 6*np.pi, periods))
    elif season_type == "irregular":
        seasonality = np.random.randint(10, 60) * np.sin(
            np.linspace(0, np.random.randint(2, 8)*np.pi, periods)
        )
    else:
        seasonality = np.zeros(periods)

    noise = np.random.normal(0, base_demand * np.random.uniform(0.05, 0.2), periods)

    y = base_demand + trend + seasonality + noise

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

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
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

    st.sidebar.success("✅ Database updated!")
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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Products", len(df))
    col2.metric("⚠️ Reorders", int(df['reorder_flag'].sum()))
    col3.metric("📊 Avg Demand", round(df['demand'].mean(), 2))
    col4.metric("📉 Avg EOQ", round(df['EOQ'].mean(), 2))

    st.divider()

    col1, col2 = st.columns(2)

    # ----------------------------
    # ✅ FIXED TOP PRODUCTS
    # ----------------------------
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

        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # STOCK VS ROP
    # ----------------------------
    with col2:
        st.subheader("Stock vs ROP")

        compare = filtered_df[['product_name', 'available_quantity', 'ROP']].head(10)

        fig2 = px.bar(
            compare,
            x='product_name',
            y=['available_quantity', 'ROP'],
            barmode='group',
            template='plotly_dark'
        )

        fig2.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # FORECAST
    # ----------------------------
    st.subheader("📈 Demand Forecast")

    selected_product = st.selectbox(
        "Select Product",
        filtered_df['product_name'].unique()
    )

    forecast = forecast_demand(selected_product, filtered_df)

    if forecast is not None:
        fig_forecast = px.line(forecast, x='ds', y='yhat', template='plotly_dark')
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("No forecast data available.")

    # ----------------------------
    # DISTRIBUTION
    # ----------------------------
    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Category Demand Share")

        cat_data = df.groupby('category')['demand'].sum().reset_index()

        fig_pie = px.pie(cat_data, names='category', values='demand',
                         template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        st.subheader("Demand Distribution")

        fig_hist = px.histogram(df, x='demand', nbins=30,
                                template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)

    # BOX
    st.subheader("Demand Variability (Box Plot)")

    fig_box = px.box(
        df,
        x='category',
        y='demand',
        color='category',
        template='plotly_dark'
    )

    st.plotly_chart(fig_box, use_container_width=True)

    # HEATMAP
    st.subheader("Category Performance Heatmap")

    heatmap_data = df.groupby('category')[['demand', 'available_quantity', 'ROP']].mean()

    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
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
    st.dataframe(filtered_df[['product_name', 'recommendation']])

# ----------------------------
# DATA
# ----------------------------
elif menu == "Data":

    st.title("📋 Inventory Data")

    st.dataframe(filtered_df)

    csv = filtered_df.to_csv(index=False)

    st.download_button("📥 Download Data", csv, "inventory.csv", "text/csv")
