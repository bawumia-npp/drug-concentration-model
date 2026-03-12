# 🔬 Pharmacokinetics Dashboard

**Professional Clinical Drug Concentration Simulation Platform**

A production-ready Streamlit application for pharmacokinetic analysis using first-order elimination models.

## 🎯 Features

- **First-Order Elimination Model**: C(t) = C₀ × exp(-k × t)
- **Single & Comparison Simulations**: Analyze one or multiple elimination rates
- **Clinical Thresholds**: MEC and MTC support with therapeutic window analysis
- **Professional Visualizations**: Interactive Plotly charts with clinical annotations
- **Persistent History**: SQLite database for simulation storage
- **Export Options**: CSV and PDF report generation
- **Hospital-Grade Interface**: Clinical dashboard design with custom CSS

## 📊 Pharmacokinetic Calculations

### Core Metrics
- **Cmax**: Maximum concentration reached
- **Tmax**: Time to reach maximum concentration
- **Half-Life**: t₁/₂ = ln(2) / k = 0.693 / k
- **AUC**: Area under the concentration-time curve using trapezoidal integration
- **Time Above MEC/MTC**: Therapeutic coverage duration

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or download the project
cd pharmacokinetics-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Running Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 🌐 Deployment to Streamlit Cloud

### Step 1: Prepare GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit: Pharmacokinetics Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pharmacokinetics-dashboard.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and `app.py` as the main file
5. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📖 Usage Guide

### HOME Tab
- Project overview
- Model formulas
- Key concepts and parameters

### SIMULATION Tab
1. Enter initial concentration (C₀)
2. Select time unit and concentration unit
3. Define simulation time range
4. Choose single or comparison mode
5. Enter elimination rate constant(s)
6. Optionally set MEC and MTC thresholds
7. Click "Run Simulation"

### RESULTS Tab
- Interactive concentration-time plot
- Pharmacokinetic metrics table
- Key metrics cards (Cmax, Tmax, Half-Life, AUC)
- Detailed time-course data table

### INTERPRETATION Tab
- Automatic clinical interpretation
- Elimination rate analysis
- Therapeutic window assessment
- Clinical significance explanation

### HISTORY Tab
- View all previous simulations
- Delete individual runs
- Clear entire history

### EXPORT Tab
- Download data as CSV
- Generate PDF clinical report
- Save simulations to persistent database

## 🔧 Technical Architecture

### Database Schema
```sql
CREATE TABLE simulations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    simulation_name TEXT,
    simulation_mode TEXT,
    c0 REAL,
    k_values TEXT,
    start_time REAL,
    end_time REAL,
    time_points INTEGER,
    time_unit TEXT,
    mec REAL,
    mtc REAL,
    cmax REAL,
    tmax REAL,
    half_life REAL,
    auc REAL,
    time_above_mec REAL,
    time_above_mtc REAL,
    interpretation TEXT,
    json_data TEXT
)
```

### Class Structure
- **DatabaseManager**: SQLite operations
- **PharmacokineticModel**: Core PK calculations
- **InterpretationEngine**: Clinical text generation
- **VisualizationEngine**: Plotly chart creation
- **ExportEngine**: CSV and PDF export

## 📊 Supported Parameters

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| C₀ | 0.1-∞ | mg/L, μg/mL, etc. | Initial concentration |
| k | 0.001-1.0 | 1/time | Elimination rate constant |
| Time | 0-∞ | hours, minutes, days | Simulation duration |
| MEC | 0-∞ | Same as C₀ | Minimum effective concentration |
| MTC | 0-∞ | Same as C₀ | Minimum toxic concentration |

## 🏥 Clinical Applications

- Drug dosing interval determination
- Therapeutic drug monitoring
- Pharmacology education
- Clinical research support
- Pharmacokinetic teaching

## ⚠️ Disclaimer

For research, training, and pharmacokinetic analysis support. **Clinical application requires independent professional validation.** This tool is designed for educational and research purposes only.

## 📝 License

This project is provided as-is for educational and professional use.

## 🤝 Support

For issues, questions, or suggestions, please send a message to emmanuelkofiavuetor@gmail.com.

---

**Version**: 2.0.0  
**Last Updated**: 2026-03-11  
**Author**: Avuetor Emmanuel