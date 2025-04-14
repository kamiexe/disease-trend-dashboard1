import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression


# --- Generate mock outbreak data ---
def generate_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    cases = np.random.poisson(lam=100, size=100) + np.linspace(0, 200, 100)
    latitudes = np.random.uniform(20.0, 30.0, 100)
    longitudes = np.random.uniform(75.0, 85.0, 100)
    data = pd.DataFrame({
        'date': dates,
        'cases': cases.astype(int),
        'latitude': latitudes,
        'longitude': longitudes
    })
    return data

# --- Predict next 7 days using linear regression ---
def predict_cases(data):
    data['days'] = (data['date'] - data['date'].min()).dt.days
    X = data[['days']]
    y = data['cases']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(data['days'].max() + 1, data['days'].max() + 8).reshape(-1, 1)
    future_preds = model.predict(future_days)
    future_dates = pd.date_range(start=data['date'].max() + pd.Timedelta(days=1), periods=7)
    return pd.DataFrame({'date': future_dates, 'predicted_cases': future_preds.astype(int)})

# --- Streamlit page setup ---
st.set_page_config(layout="wide")
st.title('Disease Trend Analysis Dashboard')
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load mock time-series data ---
data = generate_mock_data()

# --- Trend line chart ---
st.subheader(" Daily Case Trends")
fig = px.line(data, x='date', y='cases', title='Reported Cases Over Time')
st.plotly_chart(fig, use_container_width=True)

# --- Prediction chart ---
predictions = predict_cases(data)
st.subheader("Predicted Cases (Next 7 Days)")
fig2 = px.line(predictions, x='date', y='predicted_cases', title='Predicted Future Cases')
st.plotly_chart(fig2, use_container_width=True)

# --- Alert if cases exceed threshold ---
alert_threshold = 250
if any(predictions['predicted_cases'] > alert_threshold):
    st.error(" Alert: Predicted cases exceed threshold! Notify health authorities.")
else:
    st.success("No major outbreak predicted in the next 7 days.")

# --- Hotspot Map ---
st.subheader("Case Hotspot Map")
m = folium.Map(location=[25, 80], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        popup=f"{row['cases']} cases on {row['date'].date()}"
    ).add_to(marker_cluster)
st_folium(m, width=700, height=450)

# --- Disease & Region Data Summary ---
st.subheader("Disease and Region Insights")

# Mock disease + region data
disease_data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
    'Disease': ['Dengue', 'Malaria', 'Flu', 'COVID-19', 'Cholera', 'Typhoid', 'Flu', 'Dengue', 'COVID-19', 'Malaria'],
    'Cases': [120, 80, 75, 60, 40, 30, 60, 110, 90, 100]
})

# Top diseases table
st.markdown("### Top Diseases")
top_diseases = disease_data.groupby('Disease')['Cases'].sum().reset_index().sort_values(by='Cases', ascending=False)
st.dataframe(top_diseases)

# Visual: Top diseases bar chart
fig3 = px.bar(top_diseases, x='Disease', y='Cases', title='Top Diseases by Case Count', color='Cases', color_continuous_scale='Reds')
st.plotly_chart(fig3, use_container_width=True)

# Most affected regions table
st.markdown("### Most Affected Regions")
top_regions = disease_data.groupby('Region')['Cases'].sum().reset_index().sort_values(by='Cases', ascending=False)
st.dataframe(top_regions)

# Visual: Regions pie chart
fig4 = px.pie(top_regions, names='Region', values='Cases', title='Disease Burden by Region')
st.plotly_chart(fig4, use_container_width=True)
