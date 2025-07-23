"""
Explainable AI Module for MathemAI
This module provides interpretability and explanation capabilities for model predictions,
making AI decisions transparent and understandable for teachers, parents, and administrators.
"""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime


class ExplainableAI:
    """
    Main class for providing explanations of AI model predictions in MathemAI.
    Integrates SHAP and LIME for model interpretability and generates human-readable explanations.
    """
    
    def __init__(self, model, feature_names: List[str], training_data: Optional[pd.DataFrame] = None):
        """
        Initialize the Explainable AI module.
        
        Parameters:
        -----------
        model : object
            Trained model (sklearn, xgboost, or any model with predict/predict_proba methods)
        feature_names : list
            List of feature names
        training_data : pd.DataFrame, optional
            Training data for background distribution (required for SHAP)
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Feature descriptions for human-readable explanations
        self.feature_descriptions = self._initialize_feature_descriptions()
        
        # Initialize explainers if training data provided
        if training_data is not None:
            self._initialize_explainers()
    
    def _initialize_feature_descriptions(self) -> Dict[str, str]:
        """
        Initialize human-readable descriptions for each feature.
        
        Returns:
        --------
        dict
            Mapping of feature names to descriptions
        """
        descriptions = {
            # Core academic features
            'number_recognition': 'ability to recognize and identify numbers',
            'number_comparison': 'skill in comparing number magnitudes',
            'counting_skills': 'proficiency in counting sequences',
            'place_value': 'understanding of place value concepts',
            'calculation_accuracy': 'accuracy in mathematical calculations',
            'calculation_fluency': 'speed and efficiency in calculations',
            'arithmetic_facts_recall': 'memory recall of basic math facts',
            'word_problem_solving': 'ability to solve word problems',
            
            # Cognitive features
            'working_memory_score': 'working memory capacity',
            'visual_spatial_score': 'visual-spatial processing ability',
            'attention_score': 'attention and focus levels',
            
            # Composite features
            'math_ability_composite': 'overall mathematical ability',
            'cognitive_load_index': 'cognitive processing capacity',
            'number_sense_composite': 'fundamental number understanding',
            'performance_consistency': 'consistency across different skills',
            
            # Behavioral features
            'weak_areas_count': 'number of areas needing improvement',
            'strong_areas_count': 'number of areas of strength',
            'skill_imbalance': 'imbalance between different skill areas',
            'anxiety_performance_ratio': 'impact of math anxiety on performance'
        }
        
        return descriptions
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers."""
        try:
            # Initialize SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models
                if hasattr(self.model, 'booster_') or 'xgb' in str(type(self.model)).lower():
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # Use Kernel SHAP for other models
                    background = shap.sample(self.training_data, 100)
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background
                    )
            
            # Initialize LIME explainer
            if self.training_data is not None:
                self.lime_explainer = LimeTabularExplainer(
                    self.training_data.values,
                    feature_names=self.feature_names,
                    mode='classification',
                    discretize_continuous=True
                )
                
        except Exception as e:
            print(f"Warning: Could not initialize explainers: {e}")
    
    def explain_prediction(self, instance: pd.Series, method: str = 'shap') -> Dict:
        """
        Generate explanation for a single prediction.
        
        Parameters:
        -----------
        instance : pd.Series
            Single instance to explain
        method : str
            Explanation method ('shap', 'lime', or 'both')
            
        Returns:
        --------
        dict
            Comprehensive explanation including feature importance and narrative
        """
        # Get model prediction
        instance_array = instance.values.reshape(1, -1)
        prediction = self.model.predict(instance_array)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(instance_array)[0]
        else:
            probabilities = None
        
        explanation = {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'feature_importance': {},
            'top_factors': [],
            'narrative_explanation': '',
            'visual_elements': {},
            'recommendations': []
        }
        
        # Generate explanations based on method
        if method in ['shap', 'both']:
            shap_explanation = self._generate_shap_explanation(instance)
            explanation['feature_importance']['shap'] = shap_explanation
            
        if method in ['lime', 'both']:
            lime_explanation = self._generate_lime_explanation(instance)
            explanation['feature_importance']['lime'] = lime_explanation
        
        # Identify top contributing factors
        explanation['top_factors'] = self._identify_top_factors(explanation['feature_importance'])
        
        # Generate human-readable narrative
        explanation['narrative_explanation'] = self._generate_narrative(
            prediction, 
            probabilities, 
            explanation['top_factors'],
            instance
        )
        
        # Generate recommendations based on explanation
        explanation['recommendations'] = self._generate_recommendations(
            explanation['top_factors'],
            instance,
            prediction
        )
        
        return explanation
    
    def _generate_shap_explanation(self, instance: pd.Series) -> Dict:
        """Generate SHAP-based explanation."""
        if self.shap_explainer is None:
            return {}
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(instance.values.reshape(1, -1))
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For multi-class, take values for predicted class
                prediction = self.model.predict(instance.values.reshape(1, -1))[0]
                shap_values = shap_values[int(prediction)][0]
            else:
                shap_values = shap_values[0]
            
            # Create feature importance dictionary
            importance_dict = {}
            for i, feature in enumerate(self.feature_names):
                importance_dict[feature] = {
                    'shap_value': float(shap_values[i]),
                    'feature_value': float(instance.iloc[i]),
                    'impact': 'positive' if shap_values[i] > 0 else 'negative',
                    'magnitude': abs(float(shap_values[i]))
                }
            
            return importance_dict
            
        except Exception as e:
            print(f"Error generating SHAP explanation: {e}")
            return {}
    
    def _generate_lime_explanation(self, instance: pd.Series) -> Dict:
        """Generate LIME-based explanation."""
        if self.lime_explainer is None:
            return {}
        
        try:
            # Generate LIME explanation
            exp = self.lime_explainer.explain_instance(
                instance.values,
                self.model.predict_proba,
                num_features=len(self.feature_names)
            )
            
            # Convert to dictionary
            importance_dict = {}
            for feature_idx, importance in exp.as_map()[1]:
                feature_name = self.feature_names[feature_idx]
                importance_dict[feature_name] = {
                    'lime_value': float(importance),
                    'feature_value': float(instance.iloc[feature_idx]),
                    'impact': 'positive' if importance > 0 else 'negative',
                    'magnitude': abs(float(importance))
                }
            
            return importance_dict
            
        except Exception as e:
            print(f"Error generating LIME explanation: {e}")
            return {}
    
    def _identify_top_factors(self, feature_importance: Dict) -> List[Dict]:
        """Identify top contributing factors from feature importance."""
        top_factors = []
        
        # Get primary importance values (prefer SHAP if available)
        if 'shap' in feature_importance and feature_importance['shap']:
            importance_data = feature_importance['shap']
            importance_key = 'shap_value'
        elif 'lime' in feature_importance and feature_importance['lime']:
            importance_data = feature_importance['lime']
            importance_key = 'lime_value'
        else:
            return top_factors
        
        # Sort by magnitude
        sorted_features = sorted(
            importance_data.items(),
            key=lambda x: x[1]['magnitude'],
            reverse=True
        )
        
        # Get top 5 factors
        for feature, data in sorted_features[:5]:
            factor = {
                'feature': feature,
                'description': self.feature_descriptions.get(feature, feature),
                'value': data['feature_value'],
                'impact': data['impact'],
                'importance': data['magnitude'],
                'contribution': data[importance_key]
            }
            top_factors.append(factor)
        
        return top_factors
    
    def _generate_narrative(self, prediction: int, probabilities: Optional[np.ndarray], 
                          top_factors: List[Dict], instance: pd.Series) -> str:
        """
        Generate human-readable narrative explanation.
        
        Returns:
        --------
        str
            Narrative explanation suitable for teachers and parents
        """
        # Determine prediction outcome
        if prediction == 1:
            outcome = "indicates potential dyscalculia risk"
            confidence = probabilities[1] if probabilities is not None else 0.5
        else:
            outcome = "suggests no significant dyscalculia indicators"
            confidence = probabilities[0] if probabilities is not None else 0.5
        
        # Start narrative
        narrative = f"The assessment {outcome} with {confidence:.1%} confidence.\n\n"
        
        # Explain main contributing factors
        narrative += "Key factors influencing this assessment:\n\n"
        
        for i, factor in enumerate(top_factors[:3], 1):
            feature_desc = factor['description']
            value = factor['value']
            impact = factor['impact']
            
            # Create contextual explanation
            if impact == 'positive' and prediction == 1:
                direction = "increases concern"
            elif impact == 'negative' and prediction == 1:
                direction = "reduces concern"
            elif impact == 'positive' and prediction == 0:
                direction = "supports typical development"
            else:
                direction = "suggests potential difficulty"
            
            # Add context based on value
            if value < 0.3:
                level = "significantly below average"
            elif value < 0.5:
                level = "below average"
            elif value < 0.7:
                level = "average"
            elif value < 0.85:
                level = "above average"
            else:
                level = "well above average"
            
            narrative += f"{i}. The student's {feature_desc} is {level} ({value:.2f}), "
            narrative += f"which {direction}.\n"
        
        # Add overall context
        narrative += "\n"
        if prediction == 1:
            narrative += "This combination of factors suggests the student may benefit from "
            narrative += "additional support and targeted interventions in mathematical learning."
        else:
            narrative += "The student appears to be developing mathematical skills within "
            narrative += "the typical range, though continued monitoring is recommended."
        
        return narrative
    
    def _generate_recommendations(self, top_factors: List[Dict], 
                                instance: pd.Series, prediction: int) -> List[Dict]:
        """
        Generate actionable recommendations based on explanation.
        
        Returns:
        --------
        list
            List of recommendation objects
        """
        recommendations = []
        
        # Analyze weak areas from top factors
        for factor in top_factors:
            if factor['value'] < 0.5 and factor['impact'] == 'positive' and prediction == 1:
                # This is a weak area contributing to risk
                feature = factor['feature']
                
                if 'memory' in feature:
                    recommendations.append({
                        'area': 'Working Memory',
                        'priority': 'high',
                        'intervention': 'Implement memory-supporting strategies',
                        'specific_actions': [
                            'Use visual aids and manipulatives',
                            'Break problems into smaller steps',
                            'Provide written instructions alongside verbal ones'
                        ]
                    })
                
                elif 'calculation' in feature:
                    recommendations.append({
                        'area': 'Calculation Skills',
                        'priority': 'high',
                        'intervention': 'Strengthen computational foundations',
                        'specific_actions': [
                            'Practice with concrete materials',
                            'Use number lines and hundred charts',
                            'Focus on understanding before speed'
                        ]
                    })
                
                elif 'number' in feature:
                    recommendations.append({
                        'area': 'Number Sense',
                        'priority': 'high',
                        'intervention': 'Build fundamental number concepts',
                        'specific_actions': [
                            'Daily number talks',
                            'Estimation activities',
                            'Real-world number applications'
                        ]
                    })
        
        # Add general recommendations based on prediction
        if prediction == 1:
            recommendations.append({
                'area': 'General Support',
                'priority': 'medium',
                'intervention': 'Comprehensive learning support',
                'specific_actions': [
                    'Consider evaluation by learning specialist',
                    'Implement multi-sensory teaching approaches',
                    'Regular progress monitoring'
                ]
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 2))
        
        return recommendations[:4]  # Return top 4 recommendations
    
    def generate_visual_explanation(self, instance: pd.Series, save_path: Optional[str] = None) -> Dict:
        """
        Generate visual explanations of the prediction.
        
        Parameters:
        -----------
        instance : pd.Series
            Instance to explain
        save_path : str, optional
            Path to save visualizations
            
        Returns:
        --------
        dict
            Dictionary containing plot information
        """
        visual_info = {}
        
        # Generate SHAP waterfall plot
        if self.shap_explainer is not None:
            try:
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(instance.values.reshape(1, -1))
                
                # Create waterfall plot
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values[0] if not isinstance(shap_values, list) else shap_values[1][0],
                    base_values=self.shap_explainer.expected_value if not isinstance(shap_values, list) 
                                else self.shap_explainer.expected_value[1],
                    data=instance.values,
                    feature_names=self.feature_names
                ))
                
                if save_path:
                    plt.savefig(f"{save_path}_shap_waterfall.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                visual_info['shap_waterfall'] = f"{save_path}_shap_waterfall.png" if save_path else "generated"
                
            except Exception as e:
                print(f"Error generating SHAP visualization: {e}")
        
        # Generate feature importance bar plot
        explanation = self.explain_prediction(instance)
        if explanation['top_factors']:
            plt.figure(figsize=(10, 6))
            
            factors = explanation['top_factors'][:10]
            features = [f['description'] for f in factors]
            importances = [f['importance'] for f in factors]
            colors = ['red' if f['impact'] == 'negative' else 'green' for f in factors]
            
            plt.barh(features, importances, color=colors)
            plt.xlabel('Feature Importance')
            plt.title('Top Contributing Factors')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            visual_info['feature_importance'] = f"{save_path}_feature_importance.png" if save_path else "generated"
        
        return visual_info
    
    def generate_report(self, instance: pd.Series, student_id: str, 
                       report_type: str = 'teacher') -> str:
        """
        Generate a complete explanation report.
        
        Parameters:
        -----------
        instance : pd.Series
            Instance to explain
        student_id : str
            Student identifier
        report_type : str
            Type of report ('teacher', 'parent', 'administrator')
            
        Returns:
        --------
        str
            Formatted report
        """
        # Get comprehensive explanation
        explanation = self.explain_prediction(instance, method='both')
        
        # Generate report based on audience
        if report_type == 'parent':
            return self._generate_parent_report(explanation, student_id)
        elif report_type == 'administrator':
            return self._generate_admin_report(explanation, student_id)
        else:
            return self._generate_teacher_report(explanation, student_id)
    
    def _generate_teacher_report(self, explanation: Dict, student_id: str) -> str:
        """Generate detailed report for teachers."""
        report = f"""
Dyscalculia Screening Report - Teacher Version
============================================
Student ID: {student_id}
Assessment Date: {explanation['timestamp'][:10]}

ASSESSMENT OUTCOME
-----------------
{explanation['narrative_explanation']}

DETAILED ANALYSIS
----------------
"""
        
        # Add top factors with pedagogical context
        report += "\nKey Contributing Factors:\n"
        for i, factor in enumerate(explanation['top_factors'], 1):
            report += f"\n{i}. {factor['description'].title()}\n"
            report += f"   Current Level: {factor['value']:.2f}\n"
            report += f"   Impact: {factor['impact'].title()}\n"
            
            # Add teaching suggestions
            if factor['value'] < 0.5:
                report += "   Teaching Strategy: Focus on building this foundational skill\n"
            
        # Add recommendations
        report += "\nRECOMMENDED INTERVENTIONS\n"
        report += "------------------------\n"
        for rec in explanation['recommendations']:
            report += f"\n{rec['area']} (Priority: {rec['priority'].upper()})\n"
            report += f"Intervention: {rec['intervention']}\n"
            report += "Specific Actions:\n"
            for action in rec['specific_actions']:
                report += f"  • {action}\n"
        
        # Add confidence information
        if explanation['probabilities']:
            report += f"\nConfidence Level: {max(explanation['probabilities']):.1%}\n"
        
        return report
    
    def _generate_parent_report(self, explanation: Dict, student_id: str) -> str:
        """Generate simplified report for parents."""
        report = f"""
Mathematics Learning Assessment Report
====================================
Student ID: {student_id}
Date: {explanation['timestamp'][:10]}

SUMMARY
-------
"""
        
        # Simplified outcome
        prediction = explanation['prediction']
        if prediction == 1:
            report += "Your child's assessment suggests they may benefit from additional support "
            report += "in learning mathematics. This is not uncommon, and with appropriate help, "
            report += "children can develop strong math skills.\n"
        else:
            report += "Your child's mathematical development appears to be progressing typically. "
            report += "Continue to support their learning with regular practice and encouragement.\n"
        
        report += "\nAREAS TO FOCUS ON\n"
        report += "-----------------\n"
        
        # Highlight weak areas in parent-friendly language
        for factor in explanation['top_factors']:
            if factor['value'] < 0.5:
                if 'memory' in factor['feature']:
                    report += "• Remembering math facts and procedures\n"
                elif 'calculation' in factor['feature']:
                    report += "• Solving math problems accurately\n"
                elif 'number' in factor['feature']:
                    report += "• Understanding numbers and their relationships\n"
        
        report += "\nHOW YOU CAN HELP\n"
        report += "----------------\n"
        report += "• Practice math in everyday situations (cooking, shopping, games)\n"
        report += "• Be patient and positive about math learning\n"
        report += "• Work with your child's teacher on suggested activities\n"
        report += "• Consider educational apps or games that make math fun\n"
        
        if prediction == 1:
            report += "• Discuss additional support options with your child's teacher\n"
        
        return report
    
    def _generate_admin_report(self, explanation: Dict, student_id: str) -> str:
        """Generate summary report for administrators."""
        report = f"""
Dyscalculia Screening Summary - Administrative Report
===================================================
Student ID: {student_id}
Assessment Date: {explanation['timestamp'][:10]}

SCREENING RESULT: {'At Risk' if explanation['prediction'] == 1 else 'No Significant Risk'}
Confidence: {max(explanation['probabilities']):.1%} if explanation['probabilities'] else 'N/A'}

INTERVENTION REQUIREMENTS
------------------------
"""
        
        if explanation['prediction'] == 1:
            report += "☑ Additional learning support recommended\n"
            report += "☑ Regular progress monitoring advised\n"
            report += "☑ Consider specialist evaluation\n"
            
            # Resource allocation suggestions
            high_priority = [r for r in explanation['recommendations'] if r['priority'] == 'high']
            if high_priority:
                report += f"\nHigh Priority Interventions: {len(high_priority)}\n"
                report += "Estimated support hours/week: 3-5\n"
        else:
            report += "☐ Standard curriculum appropriate\n"
            report += "☐ Continue regular monitoring\n"
        
        report += "\nKEY METRICS\n"
        report += "-----------\n"
        for factor in explanation['top_factors'][:3]:
            report += f"{factor['description']}: {factor['value']:.2f}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create feature names
    feature_names = [
        'number_recognition', 'calculation_accuracy', 'working_memory_score',
        'visual_spatial_score', 'calculation_fluency', 'arithmetic_facts_recall',
        'word_problem_solving', 'attention_score', 'number_comparison', 'counting_skills'
    ]
    
    # Generate synthetic training data
    X_train = np.random.rand(n_samples, n_features)
    y_train = (X_train[:, 0] < 0.5) & (X_train[:, 1] < 0.5)  # Simple rule for demonstration
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainer
    training_df = pd.DataFrame(X_train, columns=feature_names)
    explainer = ExplainableAI(model, feature_names, training_df)
    
    # Explain a prediction
    test_instance = pd.Series(
        [0.3, 0.4, 0.6, 0.7, 0.35, 0.5, 0.4, 0.8, 0.6, 0.7],
        index=feature_names
    )
    
    # Generate explanation
    explanation = explainer.explain_prediction(test_instance)
    
    print("=== EXPLANATION SUMMARY ===")
    print(f"Prediction: {explanation['prediction']}")
    print(f"\nNarrative Explanation:")
    print(explanation['narrative_explanation'])
    
    # Generate reports
    print("\n=== TEACHER REPORT ===")
    print(explainer.generate_report(test_instance, "TEST001", "teacher"))
    
    print("\n=== PARENT REPORT ===")
    print(explainer.generate_report(test_instance, "TEST001", "parent"))