# 🎮 Platform Analytics Portfolio

**Live Demo:** [Deploy on Streamlit Cloud](https://appuct-ab-economy-balancing-7dugvujetynhmv6digqfpe.streamlit.app/)

A comprehensive interactive web application demonstrating **Product Analytics**, **A/B Testing**, and **Economy Balancing** skills for a fictional gaming/fintech platform. Built as a portfolio piece to showcase capabilities for Senior BI Analyst / Data Scientist roles at companies operating at the intersection of gaming, crypto, and digital economies.

---

## 📋 Project Purpose

This project addresses three key skill gaps:

1. **Product Analytics** - Demonstrating user behavior analysis, retention cohorts, engagement funnels, and monetization metrics
2. **A/B Testing** - Showcasing experimental design, statistical rigor, and data-driven decision making
3. **Economy Balancing** - Illustrating digital economy tuning, parameter adjustment, and outcome analysis

---

## 🚀 Features

### Module 1: Product Analytics Dashboard
- **Cohort Retention Heatmap** - Users grouped by signup week, retention tracked over 12 weeks
- **Engagement Funnel** - App Open → Session Start → Feature Use → Transaction → Repeat Transaction with drop-off rates
- **Monetization Curves** - ARPU by cohort age, segmented by user type (free vs. paying)
- **KPIs** - DAU, WAU, MAU, Stickiness, ARPU, ARPPU, Churn Rate
- **Date Range Filter** - Global filter affecting all charts

### Module 2: A/B Test Analysis Tool
- **Experiment Configuration** - Input parameters for sample size, effect size, significance level, power
- **SRM Check** - Sample Ratio Mismatch test with chi-square validation
- **Results Dashboard** - Conversion rates, lift, p-values, confidence intervals
- **Statistical Tests** - Two-proportion z-test with clear recommendations
- **Power Analysis** - Pre-test sample size calculator

### Module 3: Economy Balancer Simulator
- **Parameter Controls** - Adjust daily earn rates, item prices, tax rates, initial grants
- **Real-time Dashboard** - Total gems in circulation, velocity, Gini coefficient, purchasing power
- **Scenario Presets** - "Balanced Economy," "Runaway Inflation," "Too Stingy"
- **Alert System** - Visual warnings for hoarding or inflation risks
- **Agent-based Simulation** - 1000 simulated players over 90 days

---

## 🛠️ Technologies Used

- **Framework:** Streamlit
- **Language:** Python 3.9+
- **Data Processing:** Pandas, NumPy
- **Visualizations:** Plotly (interactive charts)
- **Statistical Analysis:** SciPy
- **Deployment:** Streamlit Community Cloud / GitHub Pages compatible

---

## 📁 Project Structure

```
product-ab-economy-balancing/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py        # Package initialization
│   ├── data_generator.py  # Synthetic data generation
│   └── analytics.py       # Analysis functions
└── data/                  # (Optional) Local data storage
```

---

## 🔧 Setup Instructions

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bernardiava/product-ab-economy-balancing.git
   cd product-ab-economy-balancing
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

---

## ☁️ Deployment to Streamlit Cloud

1. Push this repository to GitHub (already done)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click "New app" and connect your GitHub repository

4. Configure:
   - **Main file path:** `app.py`
   - **Python version:** 3.9 or higher
   - **Requirements file:** `requirements.txt`

5. Click "Deploy!"

Your app will be live at `https://your-username-product-ab-economy-balancing-app-xxxxxx.streamlit.app`

---

## 📊 Skills Demonstrated

### Product Analytics
- Cohort analysis and retention modeling
- Funnel analysis with conversion tracking
- Monetization metrics (ARPU, ARPPU, LTV concepts)
- KPI dashboard design

### A/B Testing & Experimentation
- Experimental design principles
- Sample size calculation and power analysis
- Statistical hypothesis testing (z-tests, chi-square)
- SRM (Sample Ratio Mismatch) detection
- Confidence interval interpretation
- Practical vs. statistical significance

### Economy Design & Balancing
- Virtual currency mechanics
- Supply/demand dynamics
- Sink mechanisms (taxes, depreciation)
- Wealth distribution analysis (Gini coefficient)
- Agent-based simulation
- Inflation/deflation monitoring

### Technical Skills
- Python programming
- Data manipulation with Pandas
- Interactive visualizations with Plotly
- Web application development with Streamlit
- Statistical analysis with SciPy
- Clean, modular code architecture

---

## 🎯 Use Cases

This portfolio demonstrates readiness for roles involving:

- **Product Analytics** - Understanding user behavior, retention, and monetization
- **Growth Analytics** - Running and analyzing experiments to drive product improvements
- **Game Economy Design** - Balancing virtual economies for engagement and revenue
- **Data Science** - Statistical modeling, forecasting, and insight generation
- **Business Intelligence** - Dashboard creation and KPI tracking

---

## 📝 License

MIT License - Feel free to use this code for learning and portfolio purposes.

---

## 👤 Author

**bernardiava**

GitHub: [@bernardiava](https://github.com/bernardiava)

---

*Built with ❤️ using Streamlit | Portfolio Project 2026*
