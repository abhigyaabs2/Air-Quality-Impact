# Air Quality Impact on Productivity

A machine learning project that analyzes and predicts the impact of air quality on workplace productivity using regression models.

##  Project Overview

This project explores the relationship between air quality parameters (PM2.5, PM10, CO, NO₂, O₃) and workplace productivity scores. It uses multiple regression models to predict productivity based on environmental conditions.

##  Features

- **Data Generation**: Synthetic dataset with realistic air quality patterns
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting
- **Feature Engineering**: Composite AQI, comfort scores, temporal features
- **Interactive Dashboard**: Streamlit app with real-time predictions
- **Visual Analytics**: Gauge charts, comparisons, and insights

##  Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open `air_quality.ipynb` and run all cells to:
- Generate the dataset
- Train the models
- Save model artifacts

4. **Launch the Streamlit app**
```bash
streamlit run air.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
air-quality-productivity/
├── air_quality.ipynb                 # Jupyter notebook for data analysis & model training
├── air.py                         # Streamlit deployment application
├── README.md                      # Project documentation
├── air_quality_productivity.csv   # Generated dataset (after running notebook)
├── productivity_model.pkl         # Trained model (after running notebook)
├── scaler.pkl                     # Feature scaler (after running notebook)
└── feature_columns.pkl           # Feature list (after running notebook)
```

##  Dataset Features

### Air Quality Parameters
- **PM2.5**: Fine particulate matter (µg/m³)
- **PM10**: Coarse particulate matter (µg/m³)
- **CO**: Carbon Monoxide (ppm)
- **NO₂**: Nitrogen Dioxide (ppb)
- **O₃**: Ozone (ppb)

### Environmental Parameters
- **Temperature**: Ambient temperature (°C)
- **Humidity**: Relative humidity (%)
- **Work Hours**: Daily work duration

### Temporal Features
- **Day of Week**: Monday-Friday
- **Season**: Spring, Summer, Fall, Winter

### Target Variable
- **Productivity Score**: 0-100 scale

##  Machine Learning Models

### Models Implemented
1. **Linear Regression**
   - Baseline model
   - Fast predictions
   - Interpretable coefficients

2. **Random Forest Regressor**
   - Handles non-linear relationships
   - Feature importance analysis
   - Robust to outliers

3. **Gradient Boosting Regressor**
   - Best performance
   - Sequential learning
   - High accuracy

### Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

##  Using the Streamlit App

### Main Features

1. **Interactive Sliders**: Adjust air quality parameters in real-time
2. **Productivity Gauge**: Visual representation of predicted score
3. **AQI Calculator**: Composite air quality index with health status
4. **Scenario Comparison**: Compare current vs optimal conditions
5. **Recommendations**: AI-generated suggestions for improvement
6. **Dataset Analytics**: Statistical insights and correlations

### Example Use Cases

- **Office Managers**: Optimize workplace environment
- **Health & Safety**: Monitor air quality impact
- **Researchers**: Analyze productivity patterns
- **HVAC Engineers**: Design better ventilation systems

##  Key Insights

Based on the model analysis:

1. **PM2.5 Impact**: Strong negative correlation with productivity (-0.6)
2. **Temperature Optimal**: 20-24°C range maximizes productivity
3. **Humidity Sweet Spot**: 40-60% relative humidity
4. **Work Hours**: Diminishing returns after 8 hours
5. **Combined Effect**: Multiple pollutants have cumulative impact

##  Customization

### Modify the Model

Edit the notebook to:
- Add new features
- Try different algorithms
- Adjust hyperparameters
- Change train/test split

### Customize the App

Edit `app.py` to:
- Add new visualizations
- Modify UI layout
- Include additional metrics
- Change color schemes

##  Model Performance

Typical performance metrics (varies with random seed):

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~4.5 | ~3.5 | ~0.75 |
| Random Forest | ~3.8 | ~2.9 | ~0.82 |
| Gradient Boosting | ~3.5 | ~2.7 | ~0.85 |

##  Future Enhancements

- [ ] Real-time air quality API integration
- [ ] Historical data tracking
- [ ] Multi-location comparison
- [ ] Deep learning models (LSTM, Neural Networks)
- [ ] Mobile app deployment
- [ ] PDF report generation
- [ ] Email alerts for poor conditions

##  Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

##  References

- WHO Air Quality Guidelines
- EPA Air Quality Index
- Scientific studies on productivity and air quality
- Scikit-learn documentation

##  License

This project is for educational purposes. Feel free to use and modify as needed.

##  Acknowledgments

- Synthetic data generation inspired by real-world air quality patterns
- UI design influenced by modern dashboard best practices
- Model architecture based on established regression techniques

##  Support

For questions or issues:
1. Check existing documentation
2. Review code comments
3. Open an issue on GitHub
4. Contact the development team

---

**Built using Python, Scikit-learn, and Streamlit**

*Last updated: December 2025*
