"""
Analysis module for the Platform Analytics Portfolio.
Contains functions for cohort analysis, funnel analysis, A/B testing, and economy simulation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
from datetime import timedelta


def calculate_cohort_retention(users: pd.DataFrame, 
                                events: pd.DataFrame,
                                n_weeks: int = 12) -> pd.DataFrame:
    """
    Calculate cohort retention rates by signup week.
    
    Args:
        users: DataFrame with user_id and signup_date
        events: DataFrame with user_id and event_date
        n_weeks: Number of weeks to track retention
        
    Returns:
        DataFrame with cohort week, period, and retention rate
    """
    # Assign cohort based on signup week
    users = users.copy()
    users['cohort_week'] = users['signup_date'].dt.to_period('W').dt.to_timestamp()
    
    # Get first activity date for each user
    first_activity = events.groupby('user_id')['event_date'].min().reset_index()
    first_activity.columns = ['user_id', 'first_activity']
    
    # Merge with users
    user_cohorts = users.merge(first_activity, on='user_id', how='left')
    
    # Calculate weeks since signup for each event
    events_with_cohort = events.merge(user_cohorts[['user_id', 'cohort_week']], on='user_id')
    events_with_cohort['weeks_since_signup'] = (
        (events_with_cohort['event_date'] - events_with_cohort['cohort_week']).dt.days / 7
    ).astype(int)
    
    # Filter to n_weeks
    events_with_cohort = events_with_cohort[events_with_cohort['weeks_since_signup'] < n_weeks]
    
    # Count unique users per cohort per week
    cohort_counts = events_with_cohort.groupby(['cohort_week', 'weeks_since_signup'])['user_id'].nunique().reset_index()
    cohort_counts.columns = ['cohort_week', 'period', 'active_users']
    
    # Get cohort sizes (users at week 0)
    cohort_sizes = cohort_counts[cohort_counts['period'] == 0][['cohort_week', 'active_users']]
    cohort_sizes.columns = ['cohort_week', 'cohort_size']
    
    # Calculate retention rates
    retention = cohort_counts.merge(cohort_sizes, on='cohort_week')
    retention['retention_rate'] = retention['active_users'] / retention['cohort_size']
    
    # Pivot for heatmap
    retention_pivot = retention.pivot(index='cohort_week', columns='period', values='retention_rate')
    
    return retention_pivot


def calculate_engagement_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate engagement funnel with drop-off rates.
    
    Funnel steps: App Open → Session Start → Feature Use → Transaction → Repeat Transaction
    
    Args:
        events: DataFrame with user_id, event_name, event_date
        
    Returns:
        DataFrame with funnel step, count, and drop-off rate
    """
    # Define funnel steps in order
    funnel_steps = ['app_open', 'session_start', 'feature_use', 'transaction', 'repeat_transaction']
    
    # Count unique users at each step
    funnel_data = []
    for step in funnel_steps:
        users_at_step = events[events['event_name'] == step]['user_id'].nunique()
        funnel_data.append({
            'step': step.replace('_', ' ').title(),
            'users': users_at_step
        })
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # Calculate conversion rates and drop-off
    total = funnel_df['users'].iloc[0]
    funnel_df['conversion_rate'] = funnel_df['users'] / total
    funnel_df['drop_off_rate'] = 1 - funnel_df['conversion_rate']
    
    # Calculate step-to-step drop-off
    funnel_df['step_drop_off'] = funnel_df['users'].diff() / funnel_df['users'].shift(1)
    funnel_df['step_drop_off'] = funnel_df['step_drop_off'].fillna(0)
    
    return funnel_df


def calculate_monetization_curves(users: pd.DataFrame, 
                                   events: pd.DataFrame,
                                   max_age_days: int = 90) -> pd.DataFrame:
    """
    Calculate ARPU curves by cohort age and user type.
    
    Args:
        users: DataFrame with user_id and signup_date
        events: DataFrame with user_id, event_date, revenue_amount
        max_age_days: Maximum cohort age to analyze
        
    Returns:
        DataFrame with cohort_age, user_type, and ARPU
    """
    users = users.copy()
    
    # Identify paying users
    user_revenue = events[events['revenue_amount'] > 0].groupby('user_id')['revenue_amount'].sum()
    users['is_payer'] = users['user_id'].isin(user_revenue.index)
    users['user_type'] = users['is_payer'].map({True: 'Paying', False: 'Free'})
    
    # Merge events with user signup dates
    events_with_signup = events.merge(users[['user_id', 'signup_date', 'user_type']], on='user_id')
    events_with_signup['cohort_age'] = (
        events_with_signup['event_date'] - events_with_signup['signup_date']
    ).dt.days
    
    # Filter to max_age_days
    events_with_signup = events_with_signup[events_with_signup['cohort_age'] >= 0]
    events_with_signup = events_with_signup[events_with_signup['cohort_age'] <= max_age_days]
    
    # Calculate daily revenue by cohort age and user type
    daily_revenue = events_with_signup.groupby(['cohort_age', 'user_type']).agg({
        'revenue_amount': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    daily_revenue['arpu'] = daily_revenue['revenue_amount'] / daily_revenue['user_id']
    
    return daily_revenue


def calculate_kpis(users: pd.DataFrame, 
                   events: pd.DataFrame,
                   start_date: str = None,
                   end_date: str = None) -> Dict[str, float]:
    """
    Calculate key product metrics.
    
    Args:
        users: DataFrame with user data
        events: DataFrame with event data
        start_date: Start of analysis period
        end_date: End of analysis period
        
    Returns:
        Dictionary with KPI values
    """
    if start_date is None:
        start_date = events['event_date'].min()
    if end_date is None:
        end_date = events['event_date'].max()
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter events to date range
    period_events = events[(events['event_date'] >= start_date) & 
                           (events['event_date'] <= end_date)]
    
    # DAU (average over last 7 days)
    last_7_days = end_date - timedelta(days=7)
    dau_events = period_events[period_events['event_date'] >= last_7_days]
    dau = dau_events.groupby(dau_events['event_date'].dt.date)['user_id'].nunique().mean()
    
    # WAU
    wau = period_events[period_events['event_date'] >= (end_date - timedelta(days=7))]['user_id'].nunique()
    
    # MAU
    mau = period_events[period_events['event_date'] >= (end_date - timedelta(days=30))]['user_id'].nunique()
    
    # Stickiness
    stickiness = dau / mau if mau > 0 else 0
    
    # Revenue metrics
    revenue = period_events['revenue_amount'].sum()
    payers = period_events[period_events['revenue_amount'] > 0]['user_id'].nunique()
    active_users = period_events['user_id'].nunique()
    
    arpu = revenue / active_users if active_users > 0 else 0
    arppu = revenue / payers if payers > 0 else 0
    
    # Churn rate (simplified: users who were active in prior period but not current)
    prior_start = start_date - timedelta(days=(end_date - start_date).days)
    prior_events = events[(events['event_date'] >= prior_start) & 
                          (events['event_date'] < start_date)]
    
    prior_users = set(prior_events['user_id'].unique())
    current_users = set(period_events['user_id'].unique())
    churned = prior_users - current_users
    churn_rate = len(churned) / len(prior_users) if len(prior_users) > 0 else 0
    
    return {
        'DAU': round(dau, 0),
        'WAU': round(wau, 0),
        'MAU': round(mau, 0),
        'Stickiness': round(stickiness * 100, 1),
        'ARPU': round(arpu, 2),
        'ARPPU': round(arppu, 2),
        'Churn Rate': round(churn_rate * 100, 1),
        'Total Revenue': round(revenue, 2),
        'Active Users': round(active_users, 0),
        'Paying Users': round(payers, 0)
    }


def perform_srm_check(ab_data: pd.DataFrame) -> Dict:
    """
    Perform Sample Ratio Mismatch (SRM) check using chi-square test.
    
    Args:
        ab_data: DataFrame with user_id and group assignment
        
    Returns:
        Dictionary with SRM test results
    """
    observed = ab_data['group'].value_counts()
    total = len(ab_data)
    expected = pd.Series({'control': total / 2, 'treatment': total / 2})
    
    # Chi-square test
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    srm_detected = p_value < 0.05
    
    return {
        'observed_control': int(observed.get('control', 0)),
        'observed_treatment': int(observed.get('treatment', 0)),
        'expected_control': int(expected['control']),
        'expected_treatment': int(expected['treatment']),
        'chi2_statistic': round(chi2_stat, 4),
        'p_value': round(p_value, 6),
        'srm_detected': srm_detected,
        'split_ratio': round(observed.get('treatment', 0) / observed.get('control', 1), 4)
    }


def analyze_ab_test(ab_data: pd.DataFrame, 
                    metric: str = 'retained_7d',
                    alpha: float = 0.05) -> Dict:
    """
    Analyze A/B test results with statistical tests.
    
    Args:
        ab_data: DataFrame with user_id, group, and metric
        metric: Name of the metric column to analyze
        alpha: Significance level
        
    Returns:
        Dictionary with analysis results
    """
    control = ab_data[ab_data['group'] == 'control'][metric]
    treatment = ab_data[ab_data['group'] == 'treatment'][metric]
    
    n_control = len(control)
    n_treatment = len(treatment)
    
    # Conversion rates
    p_control = control.mean()
    p_treatment = treatment.mean()
    
    # Lift
    lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
    
    # Two-proportion z-test
    p_pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    if se > 0:
        z_stat = (p_treatment - p_control) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed
    else:
        z_stat = 0
        p_value = 1.0
    
    # Confidence interval for difference
    diff = p_treatment - p_control
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control + 
                      p_treatment * (1 - p_treatment) / n_treatment)
    ci_lower = diff - stats.norm.ppf(1 - alpha/2) * se_diff
    ci_upper = diff + stats.norm.ppf(1 - alpha/2) * se_diff
    
    # Decision
    significant = p_value < alpha
    practical_significance = abs(lift) > 0.05  # 5% lift threshold
    
    if significant and lift > 0:
        recommendation = "Implement treatment - statistically significant positive lift"
    elif significant and lift < 0:
        recommendation = "Reject treatment - statistically significant negative impact"
    else:
        recommendation = "No clear winner - consider running longer or with larger sample"
    
    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'rate_control': round(p_control * 100, 2),
        'rate_treatment': round(p_treatment * 100, 2),
        'lift_percent': round(lift * 100, 2),
        'z_statistic': round(z_stat, 4),
        'p_value': round(p_value, 6),
        'ci_lower': round(ci_lower * 100, 2),
        'ci_upper': round(ci_upper * 100, 2),
        'significant': significant,
        'alpha': alpha,
        'recommendation': recommendation,
        'practical_significance': practical_significance
    }


def calculate_sample_size(baseline_rate: float, 
                          mde: float, 
                          alpha: float = 0.05, 
                          power: float = 0.8) -> int:
    """
    Calculate required sample size per variant for A/B test.
    
    Args:
        baseline_rate: Expected baseline conversion rate
        mde: Minimum detectable effect (as proportion, e.g., 0.05 for 5%)
        alpha: Significance level
        power: Statistical power (1 - beta)
        
    Returns:
        Required sample size per variant
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    p_pooled = (p1 + p2) / 2
    
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    
    n = numerator / denominator
    
    return int(np.ceil(n))


def run_economy_simulation(players: pd.DataFrame,
                            n_days: int = 90,
                            daily_earn_rate: float = 50,
                            item_prices: List[int] = None,
                            tax_rate: float = 0.05,
                            seed: int = 46) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run economy simulation over multiple days.
    
    Args:
        players: DataFrame with initial player state
        n_days: Number of days to simulate
        daily_earn_rate: Base gems earned per day
        item_prices: List of item prices
        tax_rate: Transaction tax rate (sink mechanism)
        seed: Random seed
        
    Returns:
        Tuple of (daily_economy_stats, final_player_state)
    """
    np.random.seed(seed)
    
    if item_prices is None:
        item_prices = [100, 500, 1000, 5000]
    
    players = players.copy()
    daily_stats = []
    
    for day in range(n_days):
        # Earning phase
        earn_multipliers = {'casual': 0.8, 'engaged': 1.2, 'whale': 1.5, 'spender': 1.0}
        
        for idx, player in players.iterrows():
            # Daily login probability based on player type
            login_probs = {'casual': 0.6, 'engaged': 0.85, 'whale': 0.9, 'spender': 0.75}
            
            if np.random.random() < login_probs[player['player_type']]:
                # Earn gems
                earned = int(daily_earn_rate * earn_multipliers[player['player_type']])
                bonus = np.random.choice([0, 10, 20, 50], p=[0.7, 0.15, 0.1, 0.05])
                total_earned = earned + bonus
                
                players.loc[idx, 'gems'] += total_earned
                players.loc[idx, 'total_earned'] += total_earned
                players.loc[idx, 'days_active'] += 1
        
        # Spending phase
        for idx, player in players.iterrows():
            if player['gems'] < min(item_prices):
                continue
            
            # Spending probability based on player type and gem balance
            spend_probs = {'casual': 0.3, 'engaged': 0.5, 'whale': 0.7, 'spender': 0.8}
            base_prob = spend_probs[player['player_type']]
            
            # More likely to spend if have lots of gems
            wealth_factor = min(1.0, player['gems'] / (2 * np.mean(item_prices)))
            spend_prob = base_prob * (0.5 + 0.5 * wealth_factor)
            
            if np.random.random() < spend_prob:
                # Choose affordable item
                affordable = [p for p in item_prices if p <= player['gems']]
                if affordable:
                    # Weight toward cheaper items for casual players
                    if player['player_type'] == 'casual':
                        weights = [1.0 / p for p in affordable]
                    else:
                        weights = [1.0] * len(affordable)
                    
                    price = np.random.choice(affordable, p=np.array(weights) / sum(weights))
                    
                    # Apply tax
                    tax = int(price * tax_rate)
                    actual_cost = price + tax
                    
                    if player['gems'] >= actual_cost:
                        players.loc[idx, 'gems'] -= actual_cost
                        players.loc[idx, 'total_spent'] += price
                        players.loc[idx, 'last_purchase_day'] = day
    
        # Calculate daily statistics
        total_gems = players['gems'].sum()
        total_supply = players['total_earned'].sum()
        spent_today = players['total_spent'].sum()
        
        # Gini coefficient for gem distribution
        sorted_gems = np.sort(players['gems'])
        n = len(sorted_gems)
        cumsum = np.cumsum(sorted_gems)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_gems))) / (n * total_gems) - (n + 1) / n if total_gems > 0 else 0
        
        # Velocity
        velocity = spent_today / total_supply if total_supply > 0 else 0
        
        # Average purchasing power
        avg_price = np.mean(item_prices)
        avg_purchasing_power = (players['gems'].mean() / avg_price) if avg_price > 0 else 0
        
        daily_stats.append({
            'day': day + 1,
            'total_gems': total_gems,
            'total_supply': total_supply,
            'gems_in_circulation': total_gems,
            'velocity': velocity,
            'gini_coefficient': round(gini, 4),
            'avg_purchasing_power': round(avg_purchasing_power, 2),
            'active_players': players[players['days_active'] > day].shape[0],
            'total_spent_cumulative': spent_today
        })
    
    daily_stats_df = pd.DataFrame(daily_stats)
    
    return daily_stats_df, players


def calculate_economy_health(daily_stats: pd.DataFrame) -> Dict:
    """
    Assess overall economy health from simulation results.
    
    Args:
        daily_stats: DataFrame with daily economy statistics
        
    Returns:
        Dictionary with health metrics and alerts
    """
    latest = daily_stats.iloc[-1]
    
    # Trends (last 7 days vs first 7 days)
    early_velocity = daily_stats.head(7)['velocity'].mean()
    late_velocity = daily_stats.tail(7)['velocity'].mean()
    velocity_change = (late_velocity - early_velocity) / early_velocity if early_velocity > 0 else 0
    
    early_gini = daily_stats.head(7)['gini_coefficient'].mean()
    late_gini = daily_stats.tail(7)['gini_coefficient'].mean()
    
    # Inflation indicator
    supply_growth = (latest['total_supply'] - daily_stats.iloc[0]['total_supply']) / daily_stats.iloc[0]['total_supply']
    
    # Health assessment
    alerts = []
    health_status = "Healthy"
    
    if late_velocity < 0.01:
        alerts.append("⚠️ Low velocity - players hoarding gems")
        health_status = "Deflationary Risk"
    
    if late_velocity > 0.1:
        alerts.append("⚠️ High velocity - potential inflation")
        health_status = "Inflationary Risk"
    
    if late_gini > 0.7:
        alerts.append("⚠️ High inequality - wealth concentration")
    
    if velocity_change < -0.5:
        alerts.append("⚠️ Velocity declining rapidly")
        health_status = "Deflationary Risk"
    
    if velocity_change > 0.5:
        alerts.append("⚠️ Velocity increasing rapidly")
        health_status = "Inflationary Risk"
    
    return {
        'health_status': health_status,
        'alerts': alerts,
        'final_velocity': round(late_velocity, 4),
        'velocity_trend': round(velocity_change * 100, 1),
        'final_gini': round(late_gini, 4),
        'supply_growth_percent': round(supply_growth * 100, 1),
        'avg_purchasing_power': round(latest['avg_purchasing_power'], 2)
    }


if __name__ == "__main__":
    # Test analysis functions
    from data_generator import generate_users, generate_events, generate_ab_test_data
    
    users = generate_users(500)
    events = generate_events(users, n_days=60)
    
    print("Testing cohort retention...")
    retention = calculate_cohort_retention(users, events)
    print(retention.head())
    
    print("\nTesting engagement funnel...")
    funnel = calculate_engagement_funnel(events)
    print(funnel)
    
    print("\nTesting KPIs...")
    kpis = calculate_kpis(users, events)
    print(kpis)
    
    print("\nTesting A/B analysis...")
    ab_data = generate_ab_test_data(2000)
    srm = perform_srm_check(ab_data)
    print(f"SRM Check: {srm}")
    
    results = analyze_ab_test(ab_data)
    print(f"A/B Results: {results}")
