"""
Advanced Feature Engineering Module for MathemAI
This module provides sophisticated feature engineering capabilities specifically designed
for dyscalculia detection, including domain-specific features, statistical interactions,
and intelligent feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for dyscalculia screening.
    Automatically creates domain-specific features, interactions, and selects the most relevant features.
    """
    
    def __init__(self, feature_selection_method='mutual_info', n_features=30):
        """
        Initialize the feature engineer.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection ('mutual_info', 'f_classif', 'rfe')
        n_features : int
            Number of features to select
        """
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.selected_features = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.original_features = None
        
    def create_domain_specific_features(self, df):
        """
        Create features specifically relevant to dyscalculia detection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with basic assessment scores
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional domain-specific features
        """
        df = df.copy()
        
        # Mathematical ability composite score
        if all(col in df.columns for col in ['calculation_accuracy', 'calculation_fluency', 
                                              'arithmetic_facts_recall', 'word_problem_solving']):
            df['math_ability_composite'] = (
                df['calculation_accuracy'] * 0.3 +
                df['calculation_fluency'] * 0.3 +
                df['arithmetic_facts_recall'] * 0.2 +
                df['word_problem_solving'] * 0.2
            )
        
        # Cognitive load index
        if all(col in df.columns for col in ['working_memory_score', 'visual_spatial_score']):
            df['cognitive_load_index'] = (
                df['working_memory_score'] * 0.6 +
                df['visual_spatial_score'] * 0.4
            )
            
            # Add attention score if available
            if 'attention_score' in df.columns:
                df['cognitive_load_index'] = (
                    df['working_memory_score'] * 0.5 +
                    df['visual_spatial_score'] * 0.3 +
                    df['attention_score'] * 0.2
                )
        
        # Number sense composite
        if all(col in df.columns for col in ['number_recognition', 'number_comparison', 'counting_skills']):
            df['number_sense_composite'] = (
                df['number_recognition'] * 0.4 +
                df['number_comparison'] * 0.3 +
                df['counting_skills'] * 0.3
            )
        
        # Performance consistency (standard deviation across math skills)
        math_cols = ['calculation_accuracy', 'calculation_fluency', 
                     'arithmetic_facts_recall', 'word_problem_solving']
        if all(col in df.columns for col in math_cols):
            df['performance_consistency'] = df[math_cols].std(axis=1)
            df['performance_range'] = df[math_cols].max(axis=1) - df[math_cols].min(axis=1)
        
        # Weakness indicators
        threshold_weak = 0.5
        threshold_strong = 0.8
        
        for col in math_cols:
            if col in df.columns:
                df[f'{col}_is_weak'] = (df[col] < threshold_weak).astype(int)
                df[f'{col}_is_strong'] = (df[col] > threshold_strong).astype(int)
        
        # Count of weak and strong areas
        weak_cols = [col for col in df.columns if '_is_weak' in col]
        strong_cols = [col for col in df.columns if '_is_strong' in col]
        
        if weak_cols:
            df['weak_areas_count'] = df[weak_cols].sum(axis=1)
        if strong_cols:
            df['strong_areas_count'] = df[strong_cols].sum(axis=1)
        
        # Skill imbalance score
        if 'weak_areas_count' in df.columns and 'strong_areas_count' in df.columns:
            df['skill_imbalance'] = np.abs(df['weak_areas_count'] - df['strong_areas_count'])
        
        # Speed-accuracy tradeoff (if response time available)
        if 'response_time' in df.columns and 'calculation_accuracy' in df.columns:
            df['speed_accuracy_tradeoff'] = df['calculation_accuracy'] / (df['response_time'] + 1)
            df['response_time_normalized'] = np.log1p(df['response_time'])
        
        # Mathematical anxiety impact
        if 'math_anxiety_level' in df.columns and 'math_ability_composite' in df.columns:
            # Convert anxiety to numeric if needed
            if df['math_anxiety_level'].dtype == 'object':
                anxiety_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
                df['math_anxiety_numeric'] = df['math_anxiety_level'].map(anxiety_map)
            else:
                df['math_anxiety_numeric'] = df['math_anxiety_level']
            
            df['anxiety_performance_ratio'] = df['math_anxiety_numeric'] / (df['math_ability_composite'] + 0.1)
        
        return df
    
    def create_statistical_features(self, df, numeric_cols=None):
        """
        Create statistical features from numeric columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        numeric_cols : list, optional
            List of numeric columns to use
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional statistical features
        """
        df = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any created indicator columns from statistical calculations
        numeric_cols = [col for col in numeric_cols if not any(x in col for x in ['_is_weak', '_is_strong', '_count'])]
        
        if len(numeric_cols) > 2:
            # Row-wise statistics
            df['feature_mean'] = df[numeric_cols].mean(axis=1)
            df['feature_std'] = df[numeric_cols].std(axis=1)
            df['feature_skew'] = df[numeric_cols].apply(lambda x: stats.skew(x), axis=1)
            df['feature_kurtosis'] = df[numeric_cols].apply(lambda x: stats.kurtosis(x), axis=1)
            
            # Percentile features
            df['feature_25_percentile'] = df[numeric_cols].quantile(0.25, axis=1)
            df['feature_75_percentile'] = df[numeric_cols].quantile(0.75, axis=1)
            df['feature_iqr'] = df['feature_75_percentile'] - df['feature_25_percentile']
        
        return df
    
    def create_interaction_features(self, df, important_features=None):
        """
        Create interaction features between important variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        important_features : list, optional
            List of important features for interactions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with interaction features
        """
        df = df.copy()
        
        if important_features is None:
            # Default important interactions for dyscalculia
            interactions = [
                ('working_memory_score', 'visual_spatial_score'),
                ('number_recognition', 'calculation_accuracy'),
                ('calculation_fluency', 'arithmetic_facts_recall'),
                ('cognitive_load_index', 'math_ability_composite'),
                ('number_sense_composite', 'word_problem_solving')
            ]
        else:
            # Create interactions between top features
            interactions = [(important_features[i], important_features[j]) 
                          for i in range(len(important_features)-1) 
                          for j in range(i+1, min(i+3, len(important_features)))]
        
        for feat1, feat2 in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio interaction (with small constant to avoid division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 0.001)
                
                # Difference interaction
                df[f'{feat1}_minus_{feat2}'] = np.abs(df[feat1] - df[feat2])
        
        return df
    
    def create_polynomial_features(self, df, degree=2, include_bias=False):
        """
        Create polynomial features for the most important numeric features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        degree : int
            Polynomial degree
        include_bias : bool
            Whether to include bias term
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with polynomial features
        """
        # Select only the most important numeric features to avoid feature explosion
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to core features for polynomial expansion
        core_features = ['math_ability_composite', 'cognitive_load_index', 
                        'number_sense_composite', 'performance_consistency']
        poly_features = [col for col in core_features if col in numeric_cols]
        
        if len(poly_features) > 0:
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
            poly_array = poly.fit_transform(df[poly_features])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(poly_features)
            
            # Create dataframe with polynomial features
            poly_df = pd.DataFrame(poly_array, columns=feature_names, index=df.index)
            
            # Remove the original features (they're already in df)
            poly_df = poly_df.drop(columns=poly_features, errors='ignore')
            
            # Concatenate with original dataframe
            df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def select_features(self, X, y, method=None):
        """
        Select the most relevant features using various methods.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        method : str, optional
            Feature selection method
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features
        """
        if method is None:
            method = self.feature_selection_method
        
        # Ensure no infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=self.n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit the selector
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        if method in ['mutual_info', 'f_classif']:
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices].tolist()
        else:  # RFE
            selected_features = X.columns[selector.support_].tolist()
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        return X[selected_features]
    
    def fit_transform(self, X, y):
        """
        Apply the complete feature engineering pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix
        """
        # Store original features
        self.original_features = X.columns.tolist()
        
        # 1. Create domain-specific features
        X_enhanced = self.create_domain_specific_features(X)
        
        # 2. Create statistical features
        X_enhanced = self.create_statistical_features(X_enhanced)
        
        # 3. Create interaction features
        X_enhanced = self.create_interaction_features(X_enhanced)
        
        # 4. Create polynomial features (limited to avoid explosion)
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        
        # 5. Select best features
        X_selected = self.select_features(X_enhanced, y)
        
        # 6. Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        print(f"Feature engineering complete:")
        print(f"  - Original features: {len(self.original_features)}")
        print(f"  - Enhanced features: {len(X_enhanced.columns)}")
        print(f"  - Selected features: {len(self.selected_features)}")
        
        return X_scaled
    
    def transform(self, X):
        """
        Transform new data using the fitted feature engineering pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix
        """
        # Ensure all original features are present
        missing_features = set(self.original_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Apply the same transformations
        X_enhanced = self.create_domain_specific_features(X)
        X_enhanced = self.create_statistical_features(X_enhanced)
        X_enhanced = self.create_interaction_features(X_enhanced)
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        
        # Select the same features
        X_selected = X_enhanced[self.selected_features]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        return X_scaled
    
    def get_feature_importance(self):
        """
        Get feature importance scores if available.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importance scores
        """
        if self.feature_selector is None or self.selected_features is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features
        })
        
        if hasattr(self.feature_selector, 'scores_'):
            importance_df['score'] = self.feature_selector.scores_[self.feature_selector.get_support()]
            importance_df = importance_df.sort_values('score', ascending=False)
        elif hasattr(self.feature_selector, 'ranking_'):
            importance_df['ranking'] = self.feature_selector.ranking_[self.feature_selector.support_]
            importance_df = importance_df.sort_values('ranking')
        
        return importance_df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic assessment data
    data = pd.DataFrame({
        'number_recognition': np.random.beta(2, 5, n_samples),
        'number_comparison': np.random.beta(3, 5, n_samples),
        'counting_skills': np.random.beta(2, 4, n_samples),
        'place_value': np.random.beta(2, 6, n_samples),
        'calculation_accuracy': np.random.beta(3, 5, n_samples),
        'calculation_fluency': np.random.beta(2, 5, n_samples),
        'arithmetic_facts_recall': np.random.beta(3, 6, n_samples),
        'word_problem_solving': np.random.beta(2, 7, n_samples),
        'working_memory_score': np.random.beta(4, 5, n_samples),
        'visual_spatial_score': np.random.beta(3, 4, n_samples),
        'attention_score': np.random.beta(3, 5, n_samples),
        'math_anxiety_level': np.random.choice(['low', 'medium', 'high', 'very_high'], n_samples),
        'response_time': np.random.gamma(2, 2, n_samples)
    })
    
    # Create target variable (0: no dyscalculia, 1: dyscalculia)
    # Students with low scores across multiple areas are more likely to have dyscalculia
    risk_score = (
        (data['calculation_accuracy'] < 0.4).astype(int) +
        (data['number_recognition'] < 0.4).astype(int) +
        (data['arithmetic_facts_recall'] < 0.4).astype(int) +
        (data['working_memory_score'] < 0.4).astype(int)
    )
    y = (risk_score >= 2).astype(int)
    
    # Initialize and run feature engineering
    feature_engineer = AdvancedFeatureEngineer(feature_selection_method='mutual_info', n_features=30)
    
    # Fit and transform the data
    X_transformed = feature_engineer.fit_transform(data, y)
    
    print("\nTransformed feature matrix shape:", X_transformed.shape)
    print("\nSelected features:")
    print(feature_engineer.selected_features[:10])  # Show first 10
    
    # Get feature importance
    importance_df = feature_engineer.get_feature_importance()
    if importance_df is not None:
        print("\nTop 10 most important features:")
        print(importance_df.head(10))