import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Air Quality & Productivity Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    try:
        with open('productivity_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error(" Model files not found! Please run the Jupyter notebook first to train the model.")
        st.stop()

model, scaler, feature_cols = load_model_artifacts()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('air_quality_productivity.csv')
    except FileNotFoundError:
        st.warning("Dataset not found. Generating sample data...")
        return None

df = load_data()

st.title("Air Quality Impact on Productivity Analyzer")
st.markdown("""
This application predicts workplace productivity based on air quality parameters using machine learning.
Adjust the air quality metrics below to see their impact on productivity scores.
""")


st.sidebar.header(" Input Parameters")
st.sidebar.markdown("Adjust the air quality and environmental parameters:")


with st.sidebar:
    st.subheader("Air Quality Parameters")
    
    pm25 = st.slider(
        "PM2.5 (µg/m³)",
        min_value=0.0,
        max_value=150.0,
        value=25.0,
        step=1.0,
        help="Fine particulate matter (≤2.5 micrometers)"
    )
    
    pm10 = st.slider(
        "PM10 (µg/m³)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=1.0,
        help="Coarse particulate matter (≤10 micrometers)"
    )
    
    co = st.slider(
        "CO (ppm)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Carbon Monoxide concentration"
    )
    
    no2 = st.slider(
        "NO₂ (ppb)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Nitrogen Dioxide concentration"
    )
    
    o3 = st.slider(
        "O₃ (ppb)",
        min_value=0.0,
        max_value=150.0,
        value=40.0,
        step=1.0,
        help="Ozone concentration"
    )
    
    st.subheader("Environmental Parameters")
    
    temperature = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=35.0,
        value=22.0,
        step=0.5
    )
    
    humidity = st.slider(
        "Humidity (%)",
        min_value=20.0,
        max_value=90.0,
        value=50.0,
        step=1.0
    )
    
    work_hours = st.slider(
        "Work Hours",
        min_value=4.0,
        max_value=12.0,
        value=8.0,
        step=0.5
    )
    
    st.subheader("Additional Parameters")
    
    day_of_week = st.selectbox(
        "Day of Week",
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    )
    
    season = st.selectbox(
        "Season",
        ['Spring', 'Summer', 'Fall', 'Winter']
    )


def create_input_features(pm25, pm10, co, no2, o3, temp, humidity, hours, day, season):
    
    data = {
        'pm25': pm25,
        'pm10': pm10,
        'co': co,
        'no2': no2,
        'o3': o3,
        'temperature': temp,
        'humidity': humidity,
        'work_hours': hours,
    }
    
   
    data['aqi_composite'] = (
        0.35 * pm25 + 
        0.25 * pm10 + 
        0.15 * no2 + 
        0.15 * o3 + 
        0.10 * co * 20
    )
    
    data['temp_comfort'] = max(0, min(100, 100 - 5 * abs(temp - 22)))
    data['humidity_comfort'] = max(0, min(100, 100 - 2 * abs(humidity - 50)))
    
  
    for d in ['Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        data[f'day_of_week_{d}'] = 1 if day == d else 0
    
    for s in ['Spring', 'Summer', 'Winter']:
        data[f'season_{s}'] = 1 if season == s else 0
    
    input_df = pd.DataFrame([data])
    
   
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
   
    input_df = input_df[feature_cols]
    
    return input_df


input_features = create_input_features(
    pm25, pm10, co, no2, o3, temperature, humidity, 
    work_hours, day_of_week, season
)


try:
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
except:
    prediction = model.predict(input_features)[0]


col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader(" Predicted Productivity Score")
    
   
    if prediction >= 80:
        color = "#2ecc71"
        status = "Excellent"
    elif prediction >= 60:
        color = "#f39c12"
        status = "Good"
    elif prediction >= 40:
        color = "#e67e22"
        status = "Fair"
    else:
        color = "#e74c3c"
        status = "Poor"
    
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Status: {status}", 'font': {'size': 20}},
        delta={'reference': 75, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccc'},
                {'range': [40, 60], 'color': '#fff4cc'},
                {'range': [60, 80], 'color': '#ccf2ff'},
                {'range': [80, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🌡️ Air Quality Index")
    
    aqi = 0.35 * pm25 + 0.25 * pm10 + 0.15 * no2 + 0.15 * o3 + 0.10 * co * 20
    
   
    if aqi <= 50:
        aqi_status = "Good"
        aqi_color = "#2ecc71"
    elif aqi <= 100:
        aqi_status = "Moderate"
        aqi_color = "#f39c12"
    elif aqi <= 150:
        aqi_status = "Unhealthy for Sensitive Groups"
        aqi_color = "#e67e22"
    else:
        aqi_status = "Unhealthy"
        aqi_color = "#e74c3c"
    
    st.markdown(f"""
    <div style="background-color: {aqi_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{aqi:.1f}</h2>
        <p style="color: white; margin: 5px 0 0 0; font-size: 18px;">{aqi_status}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    

    st.markdown("**Pollutant Levels:**")
    pollutants = {
        'PM2.5': (pm25, 35, '✅' if pm25 <= 35 else '⚠️'),
        'PM10': (pm10, 75, '✅' if pm10 <= 75 else '⚠️'),
        'CO': (co, 2, '✅' if co <= 2 else '⚠️'),
        'NO₂': (no2, 50, '✅' if no2 <= 50 else '⚠️'),
        'O₃': (o3, 60, '✅' if o3 <= 60 else '⚠️')
    }
    
    for pollutant, (value, threshold, icon) in pollutants.items():
        st.markdown(f"{icon} **{pollutant}**: {value:.1f} (safe: ≤{threshold})")

with col3:
    st.subheader(" Quick Stats")
    
    st.metric("Temperature", f"{temperature}°C", 
              f"{temperature - 22:+.1f}°C from optimal")
    st.metric("Humidity", f"{humidity}%",
              f"{humidity - 50:+.0f}% from optimal")
    st.metric("Work Hours", f"{work_hours} hrs")

tab1, tab2, tab3 = st.tabs(["📈 Insights", " Parameter Impact", " Dataset Overview"])

with tab1:
    st.subheader("Key Insights & Recommendations")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("** Productivity Analysis:**")
        if prediction >= 80:
            st.success("Excellent productivity conditions! Current air quality and environment are optimal.")
        elif prediction >= 60:
            st.info("Good productivity levels. Some improvements in air quality could boost performance.")
        elif prediction >= 40:
            st.warning("Fair productivity. Consider improving air quality to enhance work output.")
        else:
            st.error("Poor conditions for productivity. Immediate improvements recommended.")
        
        st.markdown("**Top Impact Factors:**")
        impacts = []
        if pm25 > 35:
            impacts.append(f"• PM2.5 is elevated ({pm25:.1f} µg/m³)")
        if pm10 > 75:
            impacts.append(f"• PM10 is elevated ({pm10:.1f} µg/m³)")
        if co > 2:
            impacts.append(f"• CO levels are high ({co:.1f} ppm)")
        if abs(temperature - 22) > 3:
            impacts.append(f"• Temperature is away from optimal (22°C)")
        if abs(humidity - 50) > 15:
            impacts.append(f"• Humidity is suboptimal (ideal: 40-60%)")
        
        if impacts:
            for impact in impacts:
                st.markdown(impact)
        else:
            st.markdown(" All parameters are within healthy ranges!")
    
    with col_b:
        st.markdown("** Recommendations:**")
        recommendations = []
        
        if pm25 > 35 or pm10 > 75:
            recommendations.append("• Use air purifiers with HEPA filters")
            recommendations.append("• Ensure proper ventilation")
        if co > 2:
            recommendations.append("• Check for CO sources and improve ventilation")
        if abs(temperature - 22) > 3:
            recommendations.append(f"• Adjust temperature to 20-24°C range")
        if humidity < 40:
            recommendations.append("• Use humidifiers to increase moisture")
        elif humidity > 60:
            recommendations.append("• Use dehumidifiers to reduce moisture")
        if work_hours > 9:
            recommendations.append("• Consider shorter work periods with breaks")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown(" Current conditions are optimal!")

with tab2:
    st.subheader("Parameter Impact on Productivity")
    
   
    scenarios = {
        'Current': prediction,
        'Optimal Air Quality': None,
        'Poor Air Quality': None,
        'Optimal Temperature': None
    }
    
  
    optimal_input = create_input_features(
        15, 30, 0.5, 20, 30, temperature, humidity,
        work_hours, day_of_week, season
    )
    try:
        optimal_scaled = scaler.transform(optimal_input)
        scenarios['Optimal Air Quality'] = model.predict(optimal_scaled)[0]
    except:
        scenarios['Optimal Air Quality'] = model.predict(optimal_input)[0]
    
  
    poor_input = create_input_features(
        100, 150, 3, 80, 100, temperature, humidity,
        work_hours, day_of_week, season
    )
    try:
        poor_scaled = scaler.transform(poor_input)
        scenarios['Poor Air Quality'] = model.predict(poor_scaled)[0]
    except:
        scenarios['Poor Air Quality'] = model.predict(poor_input)[0]
    
    
    temp_input = create_input_features(
        pm25, pm10, co, no2, o3, 22, 50,
        work_hours, day_of_week, season
    )
    try:
        temp_scaled = scaler.transform(temp_input)
        scenarios['Optimal Temperature'] = model.predict(temp_scaled)[0]
    except:
        scenarios['Optimal Temperature'] = model.predict(temp_input)[0]
    
 
    fig = px.bar(
        x=list(scenarios.keys()),
        y=list(scenarios.values()),
        labels={'x': 'Scenario', 'y': 'Productivity Score'},
        title='Productivity Under Different Scenarios',
        color=list(scenarios.values()),
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
   
    max_potential = scenarios['Optimal Air Quality']
    improvement = max_potential - prediction
    
    if improvement > 0:
        st.info(f" Potential productivity improvement: **{improvement:.1f} points** ({(improvement/prediction)*100:.1f}%) by optimizing air quality")
    else:
        st.success(" You're already at optimal productivity levels!")

with tab3:
    if df is not None:
        st.subheader("Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Avg Productivity", f"{df['productivity_score'].mean():.1f}")
        with col3:
            st.metric("Avg PM2.5", f"{df['pm25'].mean():.1f}")
        with col4:
            st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
        
      
        fig = px.histogram(
            df, 
            x='productivity_score',
            nbins=30,
            title='Distribution of Productivity Scores in Dataset',
            labels={'productivity_score': 'Productivity Score', 'count': 'Frequency'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.subheader("Feature Correlations with Productivity")
        correlations = df.select_dtypes(include=[np.number]).corr()['productivity_score'].sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.values[1:],  
            y=correlations.index[1:],
            orientation='h',
            labels={'x': 'Correlation', 'y': 'Feature'},
            title='Feature Importance',
            color=correlations.values[1:],
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset not available. Please run the Jupyter notebook first.")


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p> Air Quality Impact on Productivity Analyzer | Built with Streamlit & ML</p>
    <p style="font-size: 12px;">Data sources: Synthetic data based on real-world air quality patterns</p>
</div>
""", unsafe_allow_html=True)