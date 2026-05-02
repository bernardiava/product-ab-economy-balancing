"""
Data generation module for the Platform Analytics Portfolio.
Generates synthetic data for product analytics, A/B testing, and economy simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def generate_users(n_users: int = 5000, 
                   start_date: str = "2024-01-01",
                   end_date: str = "2024-12-01") -> pd.DataFrame:
    """
    Generate synthetic user data with signup dates, countries, and acquisition sources.
    
    Args:
        n_users: Number of users to generate
        start_date: Earliest signup date
        end_date: Latest signup date
        
    Returns:
        DataFrame with user_id, signup_date, country, acquisition_source
    """
    np.random.seed(42)
    
    # Generate signup dates with realistic distribution (more recent users)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    
    # Exponential distribution favoring more recent signups
    signup_offsets = np.random.exponential(scale=date_range/3, size=n_users)
    signup_offsets = np.clip(signup_offsets, 0, date_range)
    signup_dates = start + pd.to_timedelta(signup_offsets, unit='D')
    
    # Country distribution (realistic gaming platform distribution)
    countries = np.random.choice(
        ['US', 'UK', 'DE', 'FR', 'BR', 'JP', 'KR', 'CN', 'IN', 'RU'],
        size=n_users,
        p=[0.25, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.08, 0.09, 0.07]
    )
    
    # Acquisition source distribution
    sources = np.random.choice(
        ['organic', 'paid_social', 'paid_search', 'referral', 'influencer', 'ads'],
        size=n_users,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    )
    
    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'signup_date': signup_dates,
        'country': countries,
        'acquisition_source': sources
    })
    
    return users


def generate_events(users: pd.DataFrame,
                    n_days: int = 365,
                    event_probs: dict = None) -> pd.DataFrame:
    """
    Generate synthetic event data for users including sessions, feature usage, and transactions.
    
    Args:
        users: DataFrame with user data
        n_days: Number of days to simulate
        event_probs: Dictionary with probabilities for different events
        
    Returns:
        DataFrame with user_id, event_name, event_date, revenue_amount, session_id
    """
    if event_probs is None:
        event_probs = {
            'app_open': 0.65,
            'session_start': 0.55,
            'feature_use': 0.40,
            'transaction': 0.08,
            'repeat_transaction': 0.03
        }
    
    np.random.seed(43)
    
    events_list = []
    min_date = users['signup_date'].min()
    max_date = min_date + timedelta(days=n_days)
    
    # User activity profiles (some users are more active than others)
    user_activity_level = np.random.beta(2, 5, size=len(users))  # Skewed toward lower activity
    
    for idx, user in users.iterrows():
        user_id = user['user_id']
        signup = user['signup_date']
        
        # Days since signup that user is active
        days_active = int((max_date - signup).days)
        if days_active <= 0:
            continue
            
        # Activity level affects daily login probability
        base_prob = user_activity_level[idx]
        
        for day_offset in range(days_active):
            event_date = signup + timedelta(days=day_offset)
            
            # Retention decay over time (simplified model)
            retention_factor = np.exp(-day_offset / 60)  # ~60 day half-life
            
            # Daily activity probability
            daily_prob = base_prob * retention_factor
            
            if np.random.random() > daily_prob:
                continue
            
            # Generate events for this day
            session_id = f"sess_{user_id}_{day_offset}"
            
            # App open
            if np.random.random() < event_probs['app_open']:
                events_list.append({
                    'user_id': user_id,
                    'event_name': 'app_open',
                    'event_date': event_date,
                    'revenue_amount': 0.0,
                    'session_id': session_id
                })
            
            # Session start
            if np.random.random() < event_probs['session_start']:
                events_list.append({
                    'user_id': user_id,
                    'event_name': 'session_start',
                    'event_date': event_date,
                    'revenue_amount': 0.0,
                    'session_id': session_id
                })
                
                # Feature use (can happen multiple times per session)
                n_features = np.random.poisson(2.5)
                for _ in range(n_features):
                    if np.random.random() < event_probs['feature_use']:
                        events_list.append({
                            'user_id': user_id,
                            'event_name': 'feature_use',
                            'event_date': event_date + timedelta(minutes=np.random.randint(1, 60)),
                            'revenue_amount': 0.0,
                            'session_id': session_id
                        })
            
            # Transaction
            if np.random.random() < event_probs['transaction']:
                # Transaction amount based on user type and randomness
                base_amount = np.random.lognormal(mean=2.5, sigma=1.0)
                revenue = round(max(0.99, base_amount), 2)
                
                events_list.append({
                    'user_id': user_id,
                    'event_name': 'transaction',
                    'event_date': event_date + timedelta(hours=np.random.randint(0, 23)),
                    'revenue_amount': revenue,
                    'session_id': session_id
                })
                
                # Repeat transaction (same day)
                if np.random.random() < event_probs['repeat_transaction']:
                    revenue2 = round(np.random.lognormal(mean=2.0, sigma=0.8), 2)
                    events_list.append({
                        'user_id': user_id,
                        'event_name': 'repeat_transaction',
                        'event_date': event_date + timedelta(hours=np.random.randint(0, 23)),
                        'revenue_amount': max(0.99, revenue2),
                        'session_id': session_id
                    })
    
    events = pd.DataFrame(events_list)
    events['event_date'] = pd.to_datetime(events['event_date'])
    
    return events


def generate_ab_test_data(n_users: int = 10000,
                          baseline_retention: float = 0.35,
                          effect_size: float = 0.05,
                          seed: int = 44) -> pd.DataFrame:
    """
    Generate synthetic A/B test data for reward schedule experiment.
    
    Args:
        n_users: Total number of users in experiment
        baseline_retention: Baseline 7-day retention rate for control group
        effect_size: Expected lift in retention for treatment group
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with user_id, group, retained_7d, and other metrics
    """
    np.random.seed(seed)
    
    # Random assignment to control/treatment (50/50 split)
    groups = np.random.choice(['control', 'treatment'], size=n_users, p=[0.5, 0.5])
    
    # Calculate retention probabilities
    control_retention = baseline_retention
    treatment_retention = baseline_retention + effect_size
    
    # Generate retention outcomes
    retained = np.zeros(n_users, dtype=bool)
    for i, group in enumerate(groups):
        prob = control_retention if group == 'control' else treatment_retention
        retained[i] = np.random.random() < prob
    
    # Additional metrics for richer analysis
    sessions_7d = np.random.poisson(lam=5, size=n_users)
    sessions_7d = np.where(retained, sessions_7d + np.random.poisson(3, n_users), 
                           np.random.poisson(2, n_users))
    
    revenue_7d = np.zeros(n_users)
    payers = np.random.random(n_users) < 0.12  # 12% payer rate
    revenue_7d[payers] = np.random.lognormal(mean=3.0, sigma=1.2, size=payers.sum())
    
    ab_data = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'group': groups,
        'retained_7d': retained.astype(int),
        'sessions_7d': sessions_7d,
        'revenue_7d': np.round(revenue_7d, 2),
        'first_session_date': pd.date_range('2024-06-01', periods=n_users, freq='h')[:n_users]
    })
    
    return ab_data


def initialize_economy_simulation(n_players: int = 1000,
                                   initial_gems: int = 100,
                                   seed: int = 45) -> pd.DataFrame:
    """
    Initialize player state for economy simulation.
    
    Args:
        n_players: Number of simulated players
        initial_gems: Starting gems for new players
        seed: Random seed
        
    Returns:
        DataFrame with player states
    """
    np.random.seed(seed)
    
    # Player types affect earning and spending behavior
    player_types = np.random.choice(
        ['casual', 'engaged', 'whale', 'spender'],
        size=n_players,
        p=[0.50, 0.35, 0.05, 0.10]
    )
    
    players = pd.DataFrame({
        'player_id': range(1, n_players + 1),
        'gems': initial_gems,
        'total_earned': initial_gems,
        'total_spent': 0,
        'player_type': player_types,
        'days_active': 0,
        'last_purchase_day': -1
    })
    
    return players


if __name__ == "__main__":
    # Test data generation
    users = generate_users(1000)
    print(f"Generated {len(users)} users")
    print(users.head())
    
    events = generate_events(users.head(100), n_days=30)
    print(f"\nGenerated {len(events)} events")
    print(events.head())
    
    ab_data = generate_ab_test_data(1000)
    print(f"\nGenerated A/B test data for {len(ab_data)} users")
    print(ab_data.groupby('group')['retained_7d'].mean())
