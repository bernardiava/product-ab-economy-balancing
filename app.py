"""
Streamlit application for Platform Analytics Portfolio.
Demonstrates product analytics, A/B testing, and economy balancing skills.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import (
    generate_users, 
    generate_events, 
    generate_ab_test_data,
    initialize_economy_simulation
)
from analytics import (
    calculate_cohort_retention,
    calculate_engagement_funnel,
    calculate_monetization_curves,
    calculate_kpis,
    perform_srm_check,
    analyze_ab_test,
    calculate_sample_size,
    run_economy_simulation,
    calculate_economy_health
)


# Page configuration
st.set_page_config(
    page_title="Platform Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_product_data():
    """Load or generate product analytics data."""
    users = generate_users(n_users=5000)
    events = generate_events(users, n_days=365)
    return users, events


@st.cache_data
def load_ab_data():
    """Load or generate A/B test data."""
    return generate_ab_test_data(n_users=10000, baseline_retention=0.35, effect_size=0.05)


def render_kpi_row(kpis: dict):
    """Render a row of KPI cards."""
    cols = st.columns(5)
    
    metrics = [
        ('👥 DAU', kpis.get('DAU', 0), ''),
        ('📅 WAU', kpis.get('WAU', 0), ''),
        ('📆 MAU', kpis.get('MAU', 0), ''),
        ('📊 Stickiness', kpis.get('Stickiness', 0), '%'),
        ('💰 ARPU', kpis.get('ARPU', 0), '$'),
    ]
    
    for i, (label, value, suffix) in enumerate(metrics):
        with cols[i % 5]:
            st.metric(
                label=label,
                value=f"{value:,.0f}{suffix}" if suffix else f"{value:,.0f}"
            )


def module_product_analytics():
    """Render the Product Analytics Dashboard module."""
    st.markdown('<p class="main-header">📊 Product Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("*Demonstrating user behavior analysis, retention cohorts, engagement funnels, and monetization metrics.*")
    
    # Load data
    users, events = load_product_data()
    
    # Date range filter
    col1, col2 = st.columns([3, 1])
    with col1:
        min_date = events['event_date'].min().date()
        max_date = events['event_date'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Filter data based on date range
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, date_range
    
    filtered_events = events[
        (events['event_date'].dt.date >= start_date) & 
        (events['event_date'].dt.date <= end_date)
    ].copy()
    
    # Calculate KPIs
    kpis = calculate_kpis(users, filtered_events, 
                          pd.to_datetime(start_date), 
                          pd.to_datetime(end_date))
    
    # Display KPIs
    st.subheader("Key Performance Indicators")
    render_kpi_row(kpis)
    
    # Additional revenue metrics
    rev_cols = st.columns(3)
    with rev_cols[0]:
        st.metric("Total Revenue", f"${kpis.get('Total Revenue', 0):,.2f}")
    with rev_cols[1]:
        st.metric("Active Users", f"{kpis.get('Active Users', 0):,.0f}")
    with rev_cols[2]:
        st.metric("Paying Users", f"{kpis.get('Paying Users', 0):,.0f}")
    
    st.divider()
    
    # Cohort Retention Heatmap
    st.subheader("🔁 Cohort Retention Analysis")
    st.markdown("Users grouped by signup week, tracking retention over 12 weeks.")
    
    retention_pivot = calculate_cohort_retention(users, filtered_events, n_weeks=12)
    
    if not retention_pivot.empty:
        # Create heatmap with Plotly
        fig = px.imshow(
            retention_pivot.values,
            labels=dict(x="Week Since Signup", y="Cohort Week", color="Retention Rate"),
            x=list(range(len(retention_pivot.columns))),
            y=[d.strftime('%Y-%m-%d') for d in retention_pivot.index],
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig.update_layout(
            height=500,
            xaxis_title="Weeks Since Signup",
            yaxis_title="Cohort Start Date"
        )
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for cohort analysis in selected date range.")
    
    st.divider()
    
    # Engagement Funnel
    st.subheader("🎯 Engagement Funnel")
    st.markdown("Tracking user progression through key actions with drop-off rates.")
    
    funnel_df = calculate_engagement_funnel(filtered_events)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Funnel chart
        fig = go.Figure(go.Funnel(
            y=funnel_df['step'],
            x=funnel_df['users'],
            textposition="inside",
            textinfo="value+percent previous",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                       line=dict(width=[3, 3, 3, 3, 3]))
        ))
        fig.update_layout(
            height=400,
            title="User Engagement Funnel",
            xaxis_title="Number of Users"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drop-off table
        st.markdown("### Drop-off Analysis")
        funnel_display = funnel_df.copy()
        funnel_display['drop_off_rate'] = (funnel_display['drop_off_rate'] * 100).round(1).astype(str) + '%'
        st.dataframe(
            funnel_display[['step', 'users', 'drop_off_rate']],
            hide_index=True,
            use_container_width=True
        )
    
    st.divider()
    
    # Monetization Curves
    st.subheader("💵 Monetization Analysis")
    st.markdown("Average Revenue Per User (ARPU) by cohort age and user type.")
    
    monetization = calculate_monetization_curves(users, filtered_events, max_age_days=90)
    
    if not monetization.empty:
        tab1, tab2 = st.tabs(["ARPU Trend", "Payer vs Free"])
        
        with tab1:
            fig = px.line(
                monetization,
                x='cohort_age',
                y='arpu',
                color='user_type',
                title='ARPU by Cohort Age',
                labels={'cohort_age': 'Days Since Signup', 'arpu': 'ARPU ($)'},
                color_discrete_map={'Paying': '#2ca02c', 'Free': '#1f77b4'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            payer_summary = monetization.groupby('user_type').agg({
                'revenue_amount': 'sum',
                'user_id': 'nunique'
            }).reset_index()
            payer_summary['arpu'] = payer_summary['revenue_amount'] / payer_summary['user_id']
            
            fig = px.bar(
                payer_summary,
                x='user_type',
                y='arpu',
                title='Overall ARPU by User Type',
                labels={'user_type': 'User Type', 'arpu': 'ARPU ($)'},
                color='user_type',
                color_discrete_map={'Paying': '#2ca02c', 'Free': '#1f77b4'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No monetization data available for selected period.")


def module_ab_testing():
    """Render the A/B Test Analysis Tool module."""
    st.markdown('<p class="main-header">🧪 A/B Test Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown("*Demonstrating experimental design, statistical rigor, and data-driven decision making.*")
    
    # Load A/B test data
    ab_data = load_ab_data()
    
    # Sidebar for experiment configuration
    with st.sidebar:
        st.header("⚙️ Experiment Configuration")
        
        baseline_rate = st.slider(
            "Baseline Conversion Rate (%)",
            min_value=10.0,
            max_value=60.0,
            value=35.0,
            step=1.0
        ) / 100
        
        mde = st.slider(
            "Minimum Detectable Effect (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        ) / 100
        
        alpha = st.select_slider(
            "Significance Level (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05
        )
        
        power = st.select_slider(
            "Statistical Power (1-β)",
            options=[0.7, 0.8, 0.9],
            value=0.8
        )
        
        # Sample size calculator
        st.divider()
        st.subheader("📐 Power Analysis")
        required_n = calculate_sample_size(baseline_rate, mde, alpha, power)
        
        st.metric(
            "Required Sample Size per Variant",
            f"{required_n:,}"
        )
        
        actual_n = len(ab_data) // 2
        st.write(f"**Actual sample size:** {actual_n:,} per variant")
        
        if actual_n >= required_n:
            st.success("✅ Sample size is sufficient for detecting the specified effect.")
        else:
            st.warning(f"⚠️ Need {required_n - actual_n:,} more users per variant for adequate power.")
    
    # Main content
    st.subheader("📋 Experiment Overview")
    st.markdown("""
    **Scenario:** Testing a new reward schedule (increased daily login bonus) vs. control
    to measure impact on 7-day retention.
    """)
    
    # SRM Check
    st.subheader("🔍 Data Quality: Sample Ratio Mismatch (SRM) Check")
    srm_results = perform_srm_check(ab_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Control Group", f"{srm_results['observed_control']:,}")
    with col2:
        st.metric("Treatment Group", f"{srm_results['observed_treatment']:,}")
    with col3:
        st.metric("Split Ratio", f"{srm_results['split_ratio']:.2f}:1")
    with col4:
        if srm_results['srm_detected']:
            st.error("⚠️ SRM Detected!")
        else:
            st.success("✅ No SRM")
    
    st.write(f"**Chi-square statistic:** {srm_results['chi2_statistic']} | **p-value:** {srm_results['p_value']}")
    
    if srm_results['srm_detected']:
        st.warning("""
        ⚠️ **Warning:** Sample Ratio Mismatch detected! This could indicate:
        - Randomization issues
        - Data collection problems
        - Differential dropout between groups
        
        Proceed with caution when interpreting results.
        """)
    
    st.divider()
    
    # Results Dashboard
    st.subheader("📊 Experiment Results")
    
    results = analyze_ab_test(ab_data, metric='retained_7d', alpha=alpha)
    
    # Metrics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Control Retention",
            f"{results['rate_control']}%",
            help=f"n = {results['n_control']:,}"
        )
    
    with col2:
        st.metric(
            "Treatment Retention",
            f"{results['rate_treatment']}%",
            help=f"n = {results['n_treatment']:,}"
        )
    
    with col3:
        lift_color = "🟢" if results['lift_percent'] > 0 else "🔴" if results['lift_percent'] < 0 else "⚪"
        st.metric(
            "Lift",
            f"{lift_color} {results['lift_percent']:+.1f}%"
        )
    
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart with confidence intervals
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Control',
            x=['Retention Rate'],
            y=[results['rate_control'] / 100],
            error_y=dict(
                type='constant',
                value=np.sqrt(results['rate_control'] / 100 * (1 - results['rate_control'] / 100) / results['n_control']) * 100,
                width=0.5,
                color='#1f77b4'
            ),
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='Treatment',
            x=['Retention Rate'],
            y=[results['rate_treatment'] / 100],
            error_y=dict(
                type='constant',
                value=np.sqrt(results['rate_treatment'] / 100 * (1 - results['rate_treatment'] / 100) / results['n_treatment']) * 100,
                width=0.5,
                color='#ff7f0e'
            ),
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Conversion Rates with 95% Confidence Intervals",
            yaxis_title="Rate",
            barmode='group',
            height=400,
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistical summary
        st.markdown("### Statistical Summary")
        st.write(f"""
        - **Test:** Two-proportion z-test
        - **Z-statistic:** {results['z_statistic']}
        - **p-value:** {results['p_value']:.6f}
        - **α level:** {alpha}
        - **Significant:** {'Yes' if results['significant'] else 'No'}
        
        **95% Confidence Interval for Difference:**
        [{results['ci_lower']:.2f}%, {results['ci_upper']:.2f}%]
        """)
    
    st.divider()
    
    # Decision Output
    st.subheader("🎯 Recommendation")
    
    # Decision box styling
    if results['significant'] and results['lift_percent'] > 0:
        box_color = "#28a745"
        decision = "IMPLEMENT TREATMENT"
        icon = "✅"
    elif results['significant'] and results['lift_percent'] < 0:
        box_color = "#dc3545"
        decision = "REJECT TREATMENT"
        icon = "❌"
    else:
        box_color = "#ffc107"
        decision = "NO CLEAR WINNER"
        icon = "⏸️"
    
    st.markdown(f"""
    <div style="background-color: {box_color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">{icon} {decision}</h2>
        <p style="color: white; margin-top: 0.5rem;">{results['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Practical significance commentary
    st.markdown("### Interpretation")
    
    if results['significant']:
        stat_sig = "The result is **statistically significant** at the α={alpha} level."
    else:
        stat_sig = "The result is **not statistically significant** at the α={alpha} level."
    
    if results['practical_significance']:
        practical = "The observed lift is also **practically significant** (>5% relative improvement)."
    else:
        practical = "The observed lift may not be **practically significant** (<5% relative improvement)."
    
    st.write(f"""
    {stat_sig}
    
    {practical}
    
    **Business Context:** With {results['n_control'] + results['n_treatment']:,} users in this experiment,
    we have sufficient statistical power to detect meaningful differences. The treatment group showed
    a {results['lift_percent']:+.1f}% change in 7-day retention compared to control.
    """)
    
    # Raw data table (collapsible)
    with st.expander("📄 View Raw Metrics Table"):
        summary_table = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Sample Size': [results['n_control'], results['n_treatment']],
            'Conversion Rate': [f"{results['rate_control']}%", f"{results['rate_treatment']}%"],
            'Retained Users': [
                int(results['n_control'] * results['rate_control'] / 100),
                int(results['n_treatment'] * results['rate_treatment'] / 100)
            ]
        })
        st.dataframe(summary_table, hide_index=True, use_container_width=True)


def module_economy_balancer():
    """Render the Economy Balancer Simulator module."""
    st.markdown('<p class="main-header">⚖️ Economy Balancer Simulator</p>', unsafe_allow_html=True)
    st.markdown("*Demonstrating digital economy tuning, parameter adjustment, and outcome analysis.*")
    
    st.info("""
    **Scenario:** A platform with virtual currency ("Gems"). Players earn Gems through activities
    and spend them on items. Balance the economy to avoid inflation (hoarding) or deflation (frustration).
    """)
    
    # Parameter controls
    with st.sidebar:
        st.header("⚙️ Economy Parameters")
        
        daily_earn_rate = st.slider(
            "Daily Gem Earning Rate",
            min_value=10,
            max_value=200,
            value=50,
            step=5
        )
        
        st.subheader("Item Prices")
        item1_price = st.slider("Basic Item", 50, 500, 100, step=25)
        item2_price = st.slider("Rare Item", 200, 1000, 500, step=50)
        item3_price = st.slider("Epic Item", 500, 2000, 1000, step=100)
        item4_price = st.slider("Legendary Item", 2000, 10000, 5000, step=500)
        
        item_prices = [item1_price, item2_price, item3_price, item4_price]
        
        tax_rate = st.slider(
            "Transaction Tax Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        ) / 100
        
        initial_grant = st.slider(
            "New User Gem Grant",
            min_value=0,
            max_value=500,
            value=100,
            step=50
        )
        
        st.divider()
        
        # Scenario presets
        st.subheader("🎭 Scenario Presets")
        
        if st.button("Balanced Economy"):
            st.session_state.earn_rate = 50
            st.session_state.item_prices = [100, 500, 1000, 5000]
            st.session_state.tax_rate = 0.05
            st.session_state.initial_grant = 100
            st.rerun()
        
        if st.button("Runaway Inflation"):
            st.session_state.earn_rate = 150
            st.session_state.item_prices = [500, 2000, 5000, 10000]
            st.session_state.tax_rate = 0.0
            st.session_state.initial_grant = 500
            st.rerun()
        
        if st.button("Too Stingy"):
            st.session_state.earn_rate = 20
            st.session_state.item_prices = [200, 1000, 2000, 5000]
            st.session_state.tax_rate = 0.15
            st.session_state.initial_grant = 0
            st.rerun()
        
        # Apply preset values if they exist
        if 'earn_rate' in st.session_state:
            daily_earn_rate = st.session_state.earn_rate
            item_prices = st.session_state.item_prices
            tax_rate = st.session_state.tax_rate
            initial_grant = st.session_state.initial_grant
    
    # Run simulation
    with st.spinner("Running economy simulation..."):
        players = initialize_economy_simulation(n_players=1000, initial_gems=initial_grant)
        daily_stats, final_players = run_economy_simulation(
            players,
            n_days=90,
            daily_earn_rate=daily_earn_rate,
            item_prices=item_prices,
            tax_rate=tax_rate
        )
    
    # Economy health assessment
    health = calculate_economy_health(daily_stats)
    
    # Health status banner
    status_colors = {
        "Healthy": "#28a745",
        "Inflationary Risk": "#dc3545",
        "Deflationary Risk": "#ffc107"
    }
    
    status_color = status_colors.get(health['health_status'], "#17a2b8")
    
    st.markdown(f"""
    <div style="background-color: {status_color}; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="color: white; margin: 0;">📊 Economy Status: {health['health_status']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Alerts
    if health['alerts']:
        st.subheader("⚠️ Alerts")
        for alert in health['alerts']:
            st.warning(alert)
    
    # Key metrics
    st.subheader("📈 Economy Metrics (Day 90)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Gem Velocity",
            f"{health['final_velocity']:.4f}",
            delta=f"{health['velocity_trend']:+.1f}% trend"
        )
    
    with col2:
        st.metric(
            "Wealth Inequality (Gini)",
            f"{health['final_gini']:.3f}",
            help="0 = perfect equality, 1 = maximum inequality"
        )
    
    with col3:
        st.metric(
            "Avg Purchasing Power",
            f"{health['avg_purchasing_power']:.1f}x",
            help="Average gems held / average item price"
        )
    
    with col4:
        st.metric(
            "Supply Growth",
            f"{health['supply_growth_percent']:+.1f}%"
        )
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Total gems over time
        fig = px.area(
            daily_stats,
            x='day',
            y='gems_in_circulation',
            title='Total Gems in Circulation Over Time',
            labels={'day': 'Day', 'gems_in_circulation': 'Total Gems'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Velocity over time
        fig = px.line(
            daily_stats,
            x='day',
            y='velocity',
            title='Gem Velocity (Spending Rate / Supply)',
            labels={'day': 'Day', 'velocity': 'Velocity'}
        )
        fig.add_hline(y=0.01, line_dash="dash", line_color="orange", 
                      annotation_text="Low velocity threshold")
        fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                      annotation_text="High velocity threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gini coefficient over time
        fig = px.line(
            daily_stats,
            x='day',
            y='gini_coefficient',
            title='Wealth Distribution (Gini Coefficient)',
            labels={'day': 'Day', 'gini_coefficient': 'Gini Coefficient'}
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                      annotation_text="Moderate inequality")
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                      annotation_text="High inequality")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Purchasing power over time
        fig = px.line(
            daily_stats,
            x='day',
            y='avg_purchasing_power',
            title='Average Player Purchasing Power',
            labels={'day': 'Day', 'avg_purchasing_power': 'Purchasing Power (x avg item price)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Final distribution
    st.subheader("💎 Final Gem Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Distribution histogram
        fig = px.histogram(
            final_players,
            x='gems',
            nbins=50,
            title='Distribution of Gems Among Players',
            labels={'gems': 'Gems Held', 'count': 'Number of Players'},
            color='player_type',
            color_discrete_map={
                'casual': '#1f77b4',
                'engaged': '#2ca02c',
                'whale': '#ff7f0e',
                'spender': '#d62728'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary by player type
        type_summary = final_players.groupby('player_type').agg({
            'gems': ['mean', 'median', 'sum'],
            'total_spent': 'mean',
            'days_active': 'mean'
        }).round(2)
        type_summary.columns = ['Avg Gems', 'Median Gems', 'Total Gems', 'Avg Spent', 'Avg Active Days']
        
        st.dataframe(type_summary, use_container_width=True)
    
    # Parameter summary
    with st.expander("📝 Current Parameter Configuration"):
        st.write(f"""
        - **Daily Earn Rate:** {daily_earn_rate} gems
        - **Item Prices:** {item_prices}
        - **Tax Rate:** {tax_rate * 100:.1f}%
        - **Initial Grant:** {initial_grant} gems
        - **Simulation Duration:** 90 days
        - **Player Count:** 1000
        """)


def main():
    """Main application entry point."""
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🎮 Platform Analytics Portfolio</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Demonstrating Product Analytics, A/B Testing & Economy Balancing Skills
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Product Analytics",
        "🧪 A/B Testing",
        "⚖️ Economy Balancer"
    ])
    
    with tab1:
        module_product_analytics()
    
    with tab2:
        module_ab_testing()
    
    with tab3:
        module_economy_balancer()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with Streamlit | Portfolio Project for Senior BI Analyst / Data Scientist Role</p>
        <p>Skills Demonstrated: Python, SQL concepts, Statistical Modeling, Product Analytics, Experimental Design</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
