"""
Live Experiments Module for Real-Time A/B/n Testing.
Supports A/B tests, A/B/n tests (multiple variants), field experiments, 
randomized controlled experiments, split tests, bucket tests, and flights.

This module provides infrastructure to run actual experiments on live production data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import hashlib


class ExperimentType(Enum):
    """Types of live experiments supported."""
    AB_TEST = "ab_test"  # Two variants (A/B)
    ABN_TEST = "abn_test"  # Multiple variants (A/B/n)
    FIELD_EXPERIMENT = "field_experiment"  # Real-world setting
    RCT = "randomized_controlled_trial"  # Full RCT methodology
    SPLIT_TEST = "split_test"  # Traffic split testing
    BUCKET_TEST = "bucket_test"  # User bucketing
    FLIGHT = "flight"  # Time-based sequential testing


class AssignmentMechanism(Enum):
    """Methods for assigning users to variants."""
    COMPLETELY_RANDOMIZED = "completely_randomized"
    STRATIFIED = "stratified"
    CLUSTERED = "clustered"
    ADAPTIVE = "adaptive"  # Multi-armed bandit style


class ExperimentStatus(Enum):
    """Lifecycle states for experiments."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ANALYZING = "analyzing"
    STOPPED_EARLY = "stopped_early"


@dataclass
class Variant:
    """Represents an experimental variant."""
    name: str
    description: str = ""
    allocation_weight: float = 1.0  # For unequal splits
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.allocation_weight <= 0:
            raise ValueError("Allocation weight must be positive")


@dataclass
class MetricConfig:
    """Configuration for a metric being tracked."""
    name: str
    metric_type: str  # 'binary', 'continuous', 'count', 'ratio'
    primary: bool = False
    higher_is_better: bool = True
    minimum_effect_size: float = 0.05  # MDE
    baseline_value: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Complete configuration for a live experiment."""
    experiment_id: str
    name: str
    experiment_type: ExperimentType
    variants: List[Variant]
    metrics: List[MetricConfig]
    assignment_mechanism: AssignmentMechanism = AssignmentMechanism.COMPLETELY_RANDOMIZED
    stratification_factors: List[str] = field(default_factory=list)
    cluster_column: Optional[str] = None  # For clustered randomization
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sample_size_per_variant: int = 1000
    sequential_testing: bool = False
    alpha: float = 0.05
    power: float = 0.80
    status: ExperimentStatus = ExperimentStatus.DRAFT
    metadata: Dict = field(default_factory=dict)


class LiveExperimentEngine:
    """
    Core engine for running live A/B/n experiments on production data.
    
    Supports:
    - A/B tests (2 variants)
    - A/B/n tests (multiple variants)
    - Field experiments (real-world deployment)
    - Randomized controlled trials
    - Split/bucket tests
    - Sequential flights
    """
    
    def __init__(self, data_source: Optional[pd.DataFrame] = None):
        """
        Initialize the experiment engine.
        
        Args:
            data_source: Optional DataFrame with user/event data for analysis
        """
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.assignments: Dict[str, pd.DataFrame] = {}  # experiment_id -> assignments
        self.results_cache: Dict[str, Dict] = {}
        self.data_source = data_source
        self._assignment_seed = 42
    
    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        variants: List[Variant],
        metrics: List[MetricConfig],
        assignment_mechanism: AssignmentMechanism = AssignmentMechanism.COMPLETELY_RANDOMIZED,
        stratification_factors: List[str] = None,
        cluster_column: str = None,
        sample_size_per_variant: int = 1000,
        alpha: float = 0.05,
        power: float = 0.80,
        metadata: Dict = None
    ) -> str:
        """
        Create a new live experiment.
        
        Args:
            name: Human-readable experiment name
            experiment_type: Type of experiment (AB_TEST, ABN_TEST, etc.)
            variants: List of Variant objects
            metrics: List of MetricConfig objects
            assignment_mechanism: How to assign users
            stratification_factors: Columns to stratify on
            cluster_column: Column for clustered randomization
            sample_size_per_variant: Target sample size
            alpha: Significance level
            power: Statistical power
            metadata: Additional metadata
            
        Returns:
            experiment_id string
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        # Validate variants
        if len(variants) < 2:
            raise ValueError("Experiments must have at least 2 variants")
        
        if experiment_type == ExperimentType.AB_TEST and len(variants) != 2:
            raise ValueError("A/B tests require exactly 2 variants")
        
        # Validate metrics
        primary_metrics = [m for m in metrics if m.primary]
        if not primary_metrics:
            raise ValueError("At least one primary metric must be specified")
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            experiment_type=experiment_type,
            variants=variants,
            metrics=metrics,
            assignment_mechanism=assignment_mechanism,
            stratification_factors=stratification_factors or [],
            cluster_column=cluster_column,
            sample_size_per_variant=sample_size_per_variant,
            alpha=alpha,
            power=power,
            metadata=metadata or {}
        )
        
        self.experiments[experiment_id] = config
        return experiment_id
    
    def assign_users(
        self,
        user_ids: List[Any],
        experiment_id: str,
        user_attributes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Assign users to experimental variants using specified mechanism.
        
        Args:
            user_ids: List of user IDs to assign
            experiment_id: ID of experiment
            user_attributes: Optional DataFrame with user attributes for stratification
            
        Returns:
            DataFrame with user_id, variant_assignment, assignment_time
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Calculate allocation probabilities from weights
        weights = np.array([v.allocation_weight for v in config.variants])
        probabilities = weights / weights.sum()
        
        assignments_list = []
        
        if config.assignment_mechanism == AssignmentMechanism.COMPLETELY_RANDOMIZED:
            # Simple random assignment
            np.random.seed(self._assignment_seed)
            variant_names = [v.name for v in config.variants]
            assigned_variants = np.random.choice(
                variant_names, 
                size=len(user_ids), 
                p=probabilities
            )
            
            assignments_list = [{
                'user_id': uid,
                'variant': var,
                'assignment_time': datetime.now(),
                'assignment_method': 'completely_randomized'
            } for uid, var in zip(user_ids, assigned_variants)]
            
        elif config.assignment_mechanism == AssignmentMechanism.STRATIFIED:
            # Stratified random assignment
            if user_attributes is None:
                raise ValueError("User attributes required for stratified assignment")
            
            assignments_list = self._stratified_assignment(
                user_ids, 
                config.variants, 
                user_attributes,
                config.stratification_factors,
                probabilities
            )
            
        elif config.assignment_mechanism == AssignmentMechanism.CLUSTERED:
            # Clustered assignment (e.g., by geo, device, etc.)
            if user_attributes is None or config.cluster_column is None:
                raise ValueError("User attributes and cluster_column required for clustered assignment")
            
            assignments_list = self._clustered_assignment(
                user_ids,
                config.variants,
                user_attributes,
                config.cluster_column,
                probabilities
            )
        
        assignments_df = pd.DataFrame(assignments_list)
        self.assignments[experiment_id] = assignments_df
        
        return assignments_df
    
    def _stratified_assignment(
        self,
        user_ids: List[Any],
        variants: List[Variant],
        user_attributes: pd.DataFrame,
        stratification_factors: List[str],
        probabilities: np.ndarray
    ) -> List[Dict]:
        """Perform stratified random assignment."""
        np.random.seed(self._assignment_seed)
        
        # Create strata
        df = user_attributes.copy()
        df['user_id'] = user_ids
        strata_col = '_stratum_'
        
        if stratification_factors:
            df[strata_col] = df[stratification_factors].astype(str).agg('_'.join, axis=1)
        else:
            df[strata_col] = 'default'
        
        variant_names = [v.name for v in variants]
        assignments_list = []
        
        for stratum in df[strata_col].unique():
            stratum_users = df[df[strata_col] == stratum]['user_id'].tolist()
            n_users = len(stratum_users)
            
            # Random assignment within stratum
            assigned = np.random.choice(variant_names, size=n_users, p=probabilities)
            
            for uid, var in zip(stratum_users, assigned):
                assignments_list.append({
                    'user_id': uid,
                    'variant': var,
                    'assignment_time': datetime.now(),
                    'assignment_method': 'stratified',
                    'stratum': stratum
                })
        
        return assignments_list
    
    def _clustered_assignment(
        self,
        user_ids: List[Any],
        variants: List[Variant],
        user_attributes: pd.DataFrame,
        cluster_column: str,
        probabilities: np.ndarray
    ) -> List[Dict]:
        """Perform clustered random assignment."""
        np.random.seed(self._assignment_seed)
        
        df = user_attributes.copy()
        df['user_id'] = user_ids
        
        clusters = df[cluster_column].unique()
        variant_names = [v.name for v in variants]
        
        # Assign entire clusters to variants
        cluster_assignments = {}
        for cluster in clusters:
            cluster_assignments[cluster] = np.random.choice(variant_names, p=probabilities)
        
        assignments_list = []
        for uid in user_ids:
            user_row = df[df['user_id'] == uid]
            if len(user_row) > 0 and cluster_column in user_row.columns:
                cluster_val = user_row[cluster_column].values[0]
                variant = cluster_assignments.get(cluster_val, np.random.choice(variant_names, p=probabilities))
            else:
                variant = np.random.choice(variant_names, p=probabilities)
            
            assignments_list.append({
                'user_id': uid,
                'variant': variant,
                'assignment_time': datetime.now(),
                'assignment_method': 'clustered',
                'cluster': cluster_val if len(user_row) > 0 else None
            })
        
        return assignments_list
    
    def get_user_assignment(self, user_id: Any, experiment_id: str) -> Optional[str]:
        """
        Get the variant assignment for a specific user (for consistent experience).
        
        This enables sticky bucketing - users always see the same variant.
        """
        if experiment_id not in self.assignments:
            return None
        
        assignments = self.assignments[experiment_id]
        user_row = assignments[assignments['user_id'] == user_id]
        
        if len(user_row) == 0:
            return None
        
        return user_row['variant'].values[0]
    
    def record_event(
        self,
        experiment_id: str,
        user_id: Any,
        event_name: str,
        event_value: float = 1.0,
        timestamp: datetime = None,
        metadata: Dict = None
    ):
        """
        Record an event/metric observation for a user in the experiment.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User who triggered the event
            event_name: Name of the event (should match metric config)
            event_value: Numeric value of the event
            timestamp: When the event occurred
            metadata: Additional event metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store in a simple in-memory structure (in production, use database)
        if not hasattr(self, '_events'):
            self._events = {}
        
        key = f"{experiment_id}_{user_id}_{event_name}"
        if key not in self._events:
            self._events[key] = []
        
        self._events[key].append({
            'user_id': user_id,
            'event_name': event_name,
            'event_value': event_value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        })
    
    def analyze_results(
        self,
        experiment_id: str,
        events_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Analyze experimental results with appropriate statistical tests.
        
        Args:
            experiment_id: Experiment to analyze
            events_data: Optional DataFrame with event data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        if experiment_id not in self.assignments:
            raise ValueError(f"No assignments found for experiment {experiment_id}")
        
        assignments = self.assignments[experiment_id]
        
        # Build analysis dataset
        analysis_data = self._build_analysis_dataset(
            experiment_id, 
            assignments, 
            events_data
        )
        
        results = {
            'experiment_id': experiment_id,
            'experiment_name': config.name,
            'experiment_type': config.experiment_type.value,
            'status': config.status.value,
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_sizes': {},
            'metric_results': {},
            'overall_recommendation': None,
            'statistical_tests': []
        }
        
        # Calculate sample sizes per variant
        for variant in config.variants:
            n_users = len(analysis_data[analysis_data['variant'] == variant.name])
            results['sample_sizes'][variant.name] = n_users
        
        # Check SRM (Sample Ratio Mismatch)
        srm_result = self._check_srm(config.variants, results['sample_sizes'])
        results['srm_check'] = srm_result
        
        # Analyze each metric
        primary_significant = False
        for metric in config.metrics:
            metric_result = self._analyze_metric(
                analysis_data,
                metric,
                config.variants,
                config.alpha
            )
            results['metric_results'][metric.name] = metric_result
            
            if metric.primary and metric_result.get('significant', False):
                primary_significant = True
        
        # Overall recommendation
        results['overall_recommendation'] = self._generate_recommendation(
            results,
            primary_significant,
            config
        )
        
        self.results_cache[experiment_id] = results
        return results
    
    def _build_analysis_dataset(
        self,
        experiment_id: str,
        assignments: pd.DataFrame,
        events_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Build unified dataset for analysis."""
        # Start with assignments
        analysis_df = assignments[['user_id', 'variant']].copy()
        
        # Merge with events if provided
        if events_data is not None:
            analysis_df = analysis_df.merge(
                events_data,
                on='user_id',
                how='left'
            )
        
        # Add aggregated metrics from recorded events
        if hasattr(self, '_events'):
            # Build all metric columns at once using dict comprehension
            metric_data = {}
            
            for key, events in self._events.items():
                if key.startswith(experiment_id):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        event_name = '_'.join(parts[2:])
                        
                        # Aggregate by user
                        user_totals = {}
                        for event in events:
                            uid = event['user_id']
                            user_totals[uid] = user_totals.get(uid, 0) + event['event_value']
                        
                        # Create metric column data
                        metric_col = f"metric_{event_name}"
                        metric_data[metric_col] = analysis_df['user_id'].map(lambda uid: user_totals.get(uid, 0))
            
            # Add all columns at once
            if metric_data:
                for col, values in metric_data.items():
                    analysis_df[col] = values
        
        return analysis_df
    
    def _check_srm(
        self,
        variants: List[Variant],
        sample_sizes: Dict[str, int]
    ) -> Dict:
        """Check for Sample Ratio Mismatch."""
        total = sum(sample_sizes.values())
        
        # Expected based on allocation weights
        weights = np.array([v.allocation_weight for v in variants])
        expected_probs = weights / weights.sum()
        expected_counts = {v.name: total * p for v, p in zip(variants, expected_probs)}
        
        observed = [sample_sizes.get(v.name, 0) for v in variants]
        expected = [expected_counts[v.name] for v in variants]
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'observed': observed,
            'expected': [round(e, 2) for e in expected],
            'chi2_statistic': round(chi2_stat, 4),
            'p_value': round(p_value, 6),
            'srm_detected': p_value < 0.05,
            'interpretation': 'SRM detected - randomization may be compromised' if p_value < 0.05 else 'No SRM detected'
        }
    
    def _analyze_metric(
        self,
        data: pd.DataFrame,
        metric: MetricConfig,
        variants: List[Variant],
        alpha: float
    ) -> Dict:
        """Analyze a single metric across variants."""
        result = {
            'metric_name': metric.name,
            'metric_type': metric.metric_type,
            'is_primary': metric.primary,
            'variant_stats': {},
            'comparisons': [],
            'significant': False,
            'recommendation': None
        }
        
        # Try different column names
        possible_cols = [metric.name, f"metric_{metric.name}", f"{metric.name}_value"]
        metric_col = None
        for col in possible_cols:
            if col in data.columns:
                metric_col = col
                break
        
        if metric_col is None:
            result['error'] = f"Metric column not found: {metric.name}"
            return result
        
        # Calculate statistics per variant
        for variant in variants:
            variant_data = data[data['variant'] == variant.name][metric_col]
            
            if metric.metric_type == 'binary':
                stat = variant_data.mean()
                std = np.sqrt(stat * (1 - stat) / len(variant_data))
            else:
                stat = variant_data.mean()
                std = variant_data.std()
            
            result['variant_stats'][variant.name] = {
                'mean': round(stat, 6),
                'std': round(std, 6) if not np.isnan(std) else 0,
                'n': len(variant_data),
                'ci_lower': round(stat - 1.96 * std / np.sqrt(len(variant_data)), 6) if len(variant_data) > 0 else None,
                'ci_upper': round(stat + 1.96 * std / np.sqrt(len(variant_data)), 6) if len(variant_data) > 0 else None
            }
        
        # Statistical tests (control vs each treatment)
        control_name = variants[0].name
        control_data = data[data['variant'] == control_name][metric_col]
        
        for i, variant in enumerate(variants[1:], 1):
            treatment_data = data[data['variant'] == variant.name][metric_col]
            
            comparison = {
                'control': control_name,
                'treatment': variant.name,
                'control_mean': result['variant_stats'][control_name]['mean'],
                'treatment_mean': result['variant_stats'][variant.name]['mean'],
                'absolute_diff': round(result['variant_stats'][variant.name]['mean'] - result['variant_stats'][control_name]['mean'], 6),
                'relative_lift': None,
                'test_statistic': None,
                'p_value': None,
                'significant': False
            }
            
            # Calculate relative lift
            if result['variant_stats'][control_name]['mean'] != 0:
                comparison['relative_lift'] = round(
                    (comparison['treatment_mean'] - comparison['control_mean']) / 
                    abs(result['variant_stats'][control_name]['mean']),
                    4
                )
            
            # Appropriate statistical test
            if metric.metric_type == 'binary':
                # Two-proportion z-test
                z_stat, p_value = self._two_proportion_ztest(
                    control_data.sum(), len(control_data),
                    treatment_data.sum(), len(treatment_data)
                )
                comparison['test_statistic'] = round(z_stat, 4)
                comparison['p_value'] = round(p_value, 6)
            else:
                # Two-sample t-test (Welch's)
                t_stat, p_value = stats.ttest_ind(
                    control_data.dropna(),
                    treatment_data.dropna(),
                    equal_var=False
                )
                comparison['test_statistic'] = round(t_stat, 4)
                comparison['p_value'] = round(p_value, 6)
            
            comparison['significant'] = comparison['p_value'] < alpha
            
            if comparison['significant']:
                result['significant'] = True
            
            result['comparisons'].append(comparison)
        
        # Generate recommendation for this metric
        if result['significant']:
            best_variant = max(
                result['variant_stats'].items(),
                key=lambda x: x[1]['mean'] if metric.higher_is_better else -x[1]['mean']
            )[0]
            result['recommendation'] = f"{best_variant} performs significantly better"
        else:
            result['recommendation'] = "No statistically significant differences detected"
        
        return result
    
    def _two_proportion_ztest(
        self,
        successes_1: int, n1: int,
        successes_2: int, n2: int
    ) -> Tuple[float, float]:
        """Two-proportion z-test."""
        p1 = successes_1 / n1
        p2 = successes_2 / n2
        p_pooled = (successes_1 + successes_2) / (n1 + n2)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        if se == 0:
            return 0, 1.0
        
        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def _generate_recommendation(
        self,
        results: Dict,
        primary_significant: bool,
        config: ExperimentConfig
    ) -> str:
        """Generate overall experiment recommendation."""
        if results.get('srm_check', {}).get('srm_detected', False):
            return "⚠️ WARNING: Sample Ratio Mismatch detected. Results may be unreliable."
        
        if primary_significant:
            # Find best performing variant on primary metric
            primary_metric = [m.name for m in config.metrics if m.primary][0]
            metric_result = results['metric_results'].get(primary_metric, {})
            
            if metric_result.get('significant'):
                comparisons = metric_result.get('comparisons', [])
                best_lift = -float('inf')
                best_variant = config.variants[0].name
                
                for comp in comparisons:
                    if comp.get('relative_lift') and comp['relative_lift'] > best_lift:
                        best_lift = comp['relative_lift']
                        best_variant = comp['treatment']
                
                if best_lift > 0:
                    return f"✅ IMPLEMENT {best_variant}: Statistically significant improvement on primary metric (+{best_lift*100:.1f}%)"
                else:
                    return f"❌ REJECT treatments: Control performs best"
        
        return "⏸️ NO CLEAR WINNER: Continue experiment or iterate on variants"
    
    def start_flight(
        self,
        experiment_id: str,
        flight_name: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict:
        """
        Start a time-bounded flight within an experiment.
        
        Flights are useful for:
        - Phased rollouts
        - Learning periods
        - Avoiding novelty effects
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        flight_id = f"flight_{uuid.uuid4().hex[:8]}"
        
        flight_info = {
            'flight_id': flight_id,
            'experiment_id': experiment_id,
            'flight_name': flight_name,
            'start_time': start_time or datetime.now(),
            'end_time': end_time,
            'status': 'running' if not end_time or end_time > datetime.now() else 'scheduled',
            'created_at': datetime.now()
        }
        
        # Update experiment status
        self.experiments[experiment_id].status = ExperimentStatus.RUNNING
        if 'flights' not in self.experiments[experiment_id].metadata:
            self.experiments[experiment_id].metadata['flights'] = []
        self.experiments[experiment_id].metadata['flights'].append(flight_info)
        
        return flight_info
    
    def get_experiment_dashboard(self, experiment_id: str) -> Dict:
        """Get comprehensive dashboard data for an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        dashboard = {
            'experiment_info': {
                'id': config.experiment_id,
                'name': config.name,
                'type': config.experiment_type.value,
                'status': config.status.value,
                'created': config.start_time,
                'variants': [{'name': v.name, 'allocation': v.allocation_weight} 
                           for v in config.variants],
                'primary_metrics': [m.name for m in config.metrics if m.primary]
            },
            'live_stats': {},
            'recent_results': self.results_cache.get(experiment_id, {})
        }
        
        # Add live assignment stats
        if experiment_id in self.assignments:
            assignments = self.assignments[experiment_id]
            dashboard['live_stats'] = {
                'total_assigned': len(assignments),
                'by_variant': assignments['variant'].value_counts().to_dict(),
                'assignment_rate': 'N/A'  # Would calculate from traffic
            }
        
        return dashboard


def run_live_ab_test_example():
    """Example of running a live A/B test on real data."""
    print("=" * 70)
    print("LIVE A/B TEST EXAMPLE")
    print("=" * 70)
    
    # Initialize engine
    engine = LiveExperimentEngine()
    
    # Define variants
    variants = [
        Variant(name="control", description="Current experience", allocation_weight=1.0),
        Variant(name="treatment_a", description="New reward schedule", allocation_weight=1.0),
        Variant(name="treatment_b", description="Enhanced notifications", allocation_weight=1.0)
    ]
    
    # Define metrics
    metrics = [
        MetricConfig(
            name="retention_7d",
            metric_type="binary",
            primary=True,
            higher_is_better=True,
            baseline_value=0.35
        ),
        MetricConfig(
            name="revenue",
            metric_type="continuous",
            primary=False,
            higher_is_better=True
        ),
        MetricConfig(
            name="sessions",
            metric_type="count",
            primary=False,
            higher_is_better=True
        )
    ]
    
    # Create A/B/n experiment
    exp_id = engine.create_experiment(
        name="Q1 Retention Optimization Test",
        experiment_type=ExperimentType.ABN_TEST,
        variants=variants,
        metrics=metrics,
        assignment_mechanism=AssignmentMechanism.COMPLETELY_RANDOMIZED,
        sample_size_per_variant=5000,
        alpha=0.05,
        power=0.80,
        metadata={"team": "growth", "hypothesis": "Increased rewards improve retention"}
    )
    
    print(f"\n✅ Created experiment: {exp_id}")
    print(f"   Name: Q1 Retention Optimization Test")
    print(f"   Type: A/B/n Test (3 variants)")
    print(f"   Primary Metric: retention_7d")
    
    # Simulate live user assignments
    print("\n📊 Assigning users to variants...")
    user_ids = list(range(1, 15001))  # 15,000 users
    assignments = engine.assign_users(user_ids, exp_id)
    
    print(f"   Total users assigned: {len(assignments)}")
    print(f"   Distribution:")
    for variant in assignments['variant'].value_counts().items():
        print(f"      - {variant[0]}: {variant[1]} users")
    
    # Simulate recording events (in production, this happens in real-time)
    print("\n📈 Recording live events...")
    np.random.seed(42)
    
    # Simulate different conversion rates per variant
    retention_rates = {
        'control': 0.35,
        'treatment_a': 0.38,  # 3pp lift
        'treatment_b': 0.36   # 1pp lift
    }
    
    for _, row in assignments.iterrows():
        uid = row['user_id']
        variant = row['variant']
        
        # Simulate retention outcome
        retained = 1 if np.random.random() < retention_rates[variant] else 0
        engine.record_event(exp_id, uid, 'retention_7d', retained)
        
        # Simulate revenue
        if np.random.random() < 0.12:  # 12% payer rate
            revenue = np.random.lognormal(mean=3.0, sigma=1.2)
            engine.record_event(exp_id, uid, 'revenue', revenue)
        
        # Simulate sessions
        sessions = np.random.poisson(lam=5 if variant == 'control' else 6)
        engine.record_event(exp_id, uid, 'sessions', sessions)
    
    print("   Events recorded successfully")
    
    # Start a flight
    print("\n✈️ Starting Flight 1...")
    flight = engine.start_flight(
        exp_id,
        "Flight 1 - Initial Learning Period",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(days=7)
    )
    print(f"   Flight ID: {flight['flight_id']}")
    print(f"   Duration: 7 days")
    
    # Analyze results
    print("\n🔬 Analyzing experimental results...")
    results = engine.analyze_results(exp_id)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    
    print(f"\n📋 Experiment: {results['experiment_name']}")
    print(f"   Status: {results['status']}")
    print(f"   Analysis Time: {results['analysis_timestamp']}")
    
    print(f"\n👥 Sample Sizes:")
    for variant, size in results['sample_sizes'].items():
        print(f"   - {variant}: {size:,} users")
    
    print(f"\n🔍 SRM Check:")
    srm = results.get('srm_check', {})
    print(f"   Chi-square: {srm.get('chi2_statistic', 'N/A')}")
    print(f"   p-value: {srm.get('p_value', 'N/A')}")
    print(f"   Status: {'⚠️ SRM DETECTED' if srm.get('srm_detected') else '✅ No SRM'}")
    
    print(f"\n📊 Primary Metric Results (retention_7d):")
    primary_result = results['metric_results'].get('retention_7d', {})
    if 'variant_stats' in primary_result:
        for variant, stats_data in primary_result['variant_stats'].items():
            ci = f"[{stats_data['ci_lower']:.3f}, {stats_data['ci_upper']:.3f}]" if stats_data['ci_lower'] else "N/A"
            print(f"   {variant}: {stats_data['mean']:.3f} (95% CI: {ci})")
    
    if 'comparisons' in primary_result:
        print(f"\n   Treatment Effects:")
        for comp in primary_result['comparisons']:
            sig_marker = "✅" if comp['significant'] else "❌"
            print(f"   {sig_marker} {comp['treatment']} vs {comp['control']}:")
            print(f"       Lift: {comp['relative_lift']*100:+.1f}%")
            print(f"       p-value: {comp['p_value']:.6f}")
    
    print(f"\n🎯 Recommendation:")
    print(f"   {results['overall_recommendation']}")
    
    # Get dashboard
    print("\n📱 Dashboard Summary:")
    dashboard = engine.get_experiment_dashboard(exp_id)
    print(f"   Total Assigned: {dashboard['live_stats'].get('total_assigned', 0):,}")
    print(f"   Variants: {len(dashboard['experiment_info']['variants'])}")
    
    print("\n" + "=" * 70)
    print("✅ LIVE EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return engine, exp_id, results


if __name__ == "__main__":
    run_live_ab_test_example()
