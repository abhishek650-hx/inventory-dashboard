import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Inventory System", layout="wide")

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

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_excel("zepto_v1.xlsx")
df = df.dropna(subset=['mrp', 'quantity', 'availableQuantity'])

# ----------------------------
# CALCULATIONS
# ----------------------------
df['demand'] = df['quantity']

S = 50
df['holding_cost'] = 0.1 * df['mrp']
df['holding_cost'] = df['holding_cost'].replace(0, 0.1)

df['EOQ'] = np.sqrt((2 * df['demand'] * S) / df['holding_cost'])
df['EOQ'] = df['EOQ'].replace([np.inf, -np.inf], np.nan).fillna(0)

lead_time = 2
df['safety_stock'] = df['demand'].std() * np.sqrt(lead_time)

df['ROP'] = (df['demand'].mean() * lead_time) + df['safety_stock']
df['reorder_flag'] = df['availableQuantity'] < df['ROP']

df['recommendation'] = df.apply(
    lambda x: f"Order {max(1,int(x['EOQ']))}" if x['reorder_flag'] else "OK",
    axis=1
)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 📦 Inventory System")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Insights",
    "Data"
])

category = st.sidebar.selectbox("Category", df['Category'].dropna().unique())

filtered_df = df[df['Category'] == category]

# ----------------------------
# DASHBOARD
# ----------------------------
if menu == "Dashboard":

    st.title("📊 Inventory Dashboard")

    st.markdown("""
    <div style='background:linear-gradient(90deg,#ef4444,#7c3aed);
    padding:15px;border-radius:10px;margin-bottom:20px'>
    🚀 Real-Time Inventory Insights
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4, gap="large")

    col1.metric("📦 Products", len(df))
    col2.metric("⚠️ Reorders", int(df['reorder_flag'].sum()))
    col3.metric("❌ Out of Stock", int(df['outOfStock'].sum()))
    col4.metric("📊 Avg Demand", round(df['demand'].mean(), 2))

    st.divider()

    # ---------------- CHARTS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Demand Products")
        top_products = filtered_df.sort_values(by='demand', ascending=False).head(10)

        fig = px.bar(
            top_products,
            x='name',
            y='demand',
            color='demand',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stock vs ROP")

        compare = filtered_df[['name','availableQuantity','ROP']].head(10)

        fig2 = px.line(
            compare,
            x='name',
            y=['availableQuantity','ROP'],
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- MORE CHARTS ----------------
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Category Demand")
        cat = df.groupby('Category')['demand'].sum().reset_index()

        fig3 = px.pie(
            cat,
            names='Category',
            values='demand',
            template='plotly_dark'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Stock Distribution")

        fig4 = px.histogram(
            filtered_df,
            x='availableQuantity',
            nbins=20,
            template='plotly_dark'
        )
        st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# INSIGHTS
# ----------------------------
elif menu == "Insights":

    st.title("📊 Inventory Insights")

    alerts = filtered_df[filtered_df['reorder_flag'] == True]

    if len(alerts) > 0:
        st.markdown(f"""
        <div style='background:#7f1d1d;padding:15px;border-radius:10px'>
        ⚠️ {len(alerts)} products need reorder!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("Inventory is healthy ✅")

    st.subheader("Critical Products")

    st.dataframe(
        alerts[['name','availableQuantity','ROP','EOQ']],
        use_container_width=True
    )

    st.subheader("Recommendations")

    st.dataframe(
        filtered_df[['name','recommendation']],
        use_container_width=True
    )

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
