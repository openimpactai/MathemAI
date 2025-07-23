"""
Error Pattern Analyzer for MathemAI
This module analyzes student errors to identify patterns, misconceptions, and provide
targeted remediation strategies.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import json


class ErrorPatternAnalyzer:
    """
    Comprehensive error analysis system that identifies patterns in student mistakes,
    detects common misconceptions, and recommends targeted interventions.
    """
    
    def __init__(self):
        """Initialize the error pattern analyzer."""
        self.error_patterns = {}
        self.misconception_database = MisconceptionDatabase()
        self.error_classifier = ErrorClassifier()
        self.remediation_engine = RemediationEngine()
        self.pattern_clusters = None
        
    def analyze_student_errors(self, student_id: str, error_data: List[Dict]) -> Dict:
        """
        Analyze a student's errors to identify patterns and misconceptions.
        
        Parameters:
        -----------
        student_id : str
            Unique student identifier
        error_data : list
            List of error records containing questions, answers, and context
            
        Returns:
        --------
        dict
            Comprehensive error analysis report
        """
        analysis_report = {
            'student_id': student_id,
            'analysis_date': datetime.now().isoformat(),
            'total_errors': len(error_data),
            'error_types': defaultdict(int),
            'error_patterns': [],
            'misconceptions': [],
            'severity_assessment': {},
            'recommendations': [],
            'progress_indicators': {}
        }
        
        # Classify each error
        classified_errors = []
        for error in error_data:
            classification = self.error_classifier.classify_error(error)
            classified_errors.append(classification)
            analysis_report['error_types'][classification['primary_type']] += 1
        
        # Find error patterns
        patterns = self._identify_error_patterns(classified_errors)
        analysis_report['error_patterns'] = patterns
        
        # Detect misconceptions
        misconceptions = self._detect_misconceptions(classified_errors, error_data)
        analysis_report['misconceptions'] = misconceptions
        
        # Assess severity
        analysis_report['severity_assessment'] = self._assess_error_severity(
            analysis_report['error_types'],
            patterns,
            misconceptions
        )
        
        # Generate recommendations
        analysis_report['recommendations'] = self.remediation_engine.generate_recommendations(
            error_types=analysis_report['error_types'],
            patterns=patterns,
            misconceptions=misconceptions,
            severity=analysis_report['severity_assessment']
        )
        
        # Calculate progress indicators
        analysis_report['progress_indicators'] = self._calculate_progress_indicators(
            classified_errors,
            error_data
        )
        
        return analysis_report
    
    def _identify_error_patterns(self, classified_errors: List[Dict]) -> List[Dict]:
        """
        Identify recurring patterns in errors using clustering and statistical analysis.
        
        Returns:
        --------
        list
            List of identified error patterns
        """
        patterns = []
        
        # Group errors by type and context
        error_groups = defaultdict(list)
        for error in classified_errors:
            key = (error['primary_type'], error.get('skill_area', 'general'))
            error_groups[key].append(error)
        
        # Analyze each group for patterns
        for (error_type, skill_area), errors in error_groups.items():
            if len(errors) >= 3:  # Minimum errors to establish pattern
                pattern = {
                    'type': error_type,
                    'skill_area': skill_area,
                    'frequency': len(errors),
                    'consistency': self._calculate_consistency(errors),
                    'examples': errors[:3],  # Sample examples
                    'triggers': self._identify_triggers(errors)
                }
                
                # Determine if this is a significant pattern
                if pattern['frequency'] >= 5 or pattern['consistency'] > 0.7:
                    patterns.append(pattern)
        
        # Sort by significance (frequency * consistency)
        patterns.sort(key=lambda x: x['frequency'] * x['consistency'], reverse=True)
        
        return patterns
    
    def _calculate_consistency(self, errors: List[Dict]) -> float:
        """
        Calculate how consistent an error pattern is.
        
        Returns:
        --------
        float
            Consistency score (0-1)
        """
        if len(errors) < 2:
            return 0.0
        
        # Check similarity of error contexts
        contexts = [e.get('context', {}) for e in errors]
        
        # Simple consistency based on repeated similar contexts
        context_features = []
        for ctx in contexts:
            features = [
                ctx.get('difficulty', 0),
                ctx.get('time_pressure', 0),
                ctx.get('problem_type', '')
            ]
            context_features.append(str(features))
        
        # Count most common context
        context_counter = Counter(context_features)
        most_common_count = context_counter.most_common(1)[0][1]
        
        return most_common_count / len(errors)
    
    def _identify_triggers(self, errors: List[Dict]) -> List[str]:
        """
        Identify common triggers or conditions that lead to errors.
        
        Returns:
        --------
        list
            List of identified triggers
        """
        triggers = []
        
        # Analyze contexts for common factors
        all_contexts = [e.get('context', {}) for e in errors]
        
        # Check for time pressure
        time_pressured = sum(1 for ctx in all_contexts if ctx.get('time_pressure', 0) > 0.7)
        if time_pressured / len(errors) > 0.6:
            triggers.append('time_pressure')
        
        # Check for specific problem types
        problem_types = [ctx.get('problem_type', '') for ctx in all_contexts if ctx.get('problem_type')]
        if problem_types:
            type_counter = Counter(problem_types)
            common_type = type_counter.most_common(1)[0]
            if common_type[1] / len(errors) > 0.5:
                triggers.append(f'problem_type:{common_type[0]}')
        
        # Check for difficulty levels
        difficulties = [ctx.get('difficulty', 0) for ctx in all_contexts]
        avg_difficulty = np.mean(difficulties) if difficulties else 0
        if avg_difficulty > 0.7:
            triggers.append('high_difficulty')
        elif avg_difficulty < 0.3:
            triggers.append('low_difficulty')
        
        return triggers
    
    def _detect_misconceptions(self, classified_errors: List[Dict], 
                             original_errors: List[Dict]) -> List[Dict]:
        """
        Detect underlying misconceptions from error patterns.
        
        Returns:
        --------
        list
            List of detected misconceptions
        """
        misconceptions = []
        
        # Check each error against known misconception patterns
        for i, error in enumerate(classified_errors):
            original = original_errors[i] if i < len(original_errors) else {}
            
            # Check for specific misconception indicators
            misconception = self.misconception_database.check_misconception(
                error_type=error['primary_type'],
                student_answer=original.get('student_answer', ''),
                correct_answer=original.get('correct_answer', ''),
                question=original.get('question', '')
            )
            
            if misconception:
                # Check if we've already identified this misconception
                existing = next((m for m in misconceptions if m['type'] == misconception['type']), None)
                if existing:
                    existing['frequency'] += 1
                    existing['examples'].append(original)
                else:
                    misconception['frequency'] = 1
                    misconception['examples'] = [original]
                    misconceptions.append(misconception)
        
        # Sort by frequency
        misconceptions.sort(key=lambda x: x['frequency'], reverse=True)
        
        return misconceptions
    
    def _assess_error_severity(self, error_types: Dict[str, int], 
                             patterns: List[Dict], 
                             misconceptions: List[Dict]) -> Dict:
        """
        Assess the overall severity of error patterns.
        
        Returns:
        --------
        dict
            Severity assessment
        """
        severity = {
            'overall': 'low',
            'immediate_intervention_needed': False,
            'areas_of_concern': [],
            'risk_factors': []
        }
        
        # Calculate severity score
        severity_score = 0
        
        # Factor 1: Total error volume
        total_errors = sum(error_types.values())
        if total_errors > 20:
            severity_score += 2
            severity['risk_factors'].append('high_error_volume')
        elif total_errors > 10:
            severity_score += 1
        
        # Factor 2: Presence of fundamental misconceptions
        fundamental_misconceptions = [m for m in misconceptions 
                                    if m.get('severity', 'low') in ['high', 'critical']]
        if fundamental_misconceptions:
            severity_score += 3
            severity['risk_factors'].append('fundamental_misconceptions')
            severity['areas_of_concern'].extend([m['type'] for m in fundamental_misconceptions])
        
        # Factor 3: Persistent patterns
        persistent_patterns = [p for p in patterns if p['consistency'] > 0.8]
        if persistent_patterns:
            severity_score += 2
            severity['risk_factors'].append('persistent_error_patterns')
        
        # Factor 4: Error diversity (many different types of errors)
        if len(error_types) > 5:
            severity_score += 1
            severity['risk_factors'].append('diverse_error_types')
        
        # Determine overall severity
        if severity_score >= 6:
            severity['overall'] = 'critical'
            severity['immediate_intervention_needed'] = True
        elif severity_score >= 4:
            severity['overall'] = 'high'
            severity['immediate_intervention_needed'] = True
        elif severity_score >= 2:
            severity['overall'] = 'moderate'
        else:
            severity['overall'] = 'low'
        
        return severity
    
    def _calculate_progress_indicators(self, classified_errors: List[Dict], 
                                     error_data: List[Dict]) -> Dict:
        """
        Calculate indicators that can track progress over time.
        
        Returns:
        --------
        dict
            Progress tracking indicators
        """
        indicators = {
            'error_rate_by_difficulty': {},
            'time_to_error': [],
            'self_correction_rate': 0,
            'error_clustering_coefficient': 0
        }
        
        # Error rate by difficulty
        difficulty_groups = defaultdict(lambda: {'total': 0, 'errors': 0})
        for error in error_data:
            difficulty = error.get('context', {}).get('difficulty', 'medium')
            difficulty_groups[difficulty]['total'] += 1
            difficulty_groups[difficulty]['errors'] += 1
        
        for diff, counts in difficulty_groups.items():
            indicators['error_rate_by_difficulty'][diff] = counts['errors'] / max(1, counts['total'])
        
        # Time to error (if available)
        for error in error_data:
            if 'time_taken' in error:
                indicators['time_to_error'].append(error['time_taken'])
        
        # Self-correction rate (if data available)
        corrections = sum(1 for e in error_data if e.get('self_corrected', False))
        indicators['self_correction_rate'] = corrections / max(1, len(error_data))
        
        # Error clustering coefficient (how clustered errors are)
        if len(classified_errors) > 1:
            error_sequence = [e['primary_type'] for e in classified_errors]
            clusters = self._calculate_clustering(error_sequence)
            indicators['error_clustering_coefficient'] = clusters
        
        return indicators
    
    def _calculate_clustering(self, sequence: List[str]) -> float:
        """
        Calculate how clustered errors are in sequence.
        
        Returns:
        --------
        float
            Clustering coefficient (0-1)
        """
        if len(sequence) < 2:
            return 0.0
        
        # Count consecutive same errors
        consecutive_count = 0
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                consecutive_count += 1
        
        # Normalize by maximum possible consecutive errors
        max_consecutive = len(sequence) - 1
        return consecutive_count / max_consecutive
    
    def generate_error_report(self, analysis: Dict, format: str = 'detailed') -> str:
        """
        Generate a formatted error analysis report.
        
        Parameters:
        -----------
        analysis : dict
            Error analysis results
        format : str
            Report format ('detailed', 'summary', 'actionable')
            
        Returns:
        --------
        str
            Formatted report
        """
        if format == 'summary':
            return self._generate_summary_report(analysis)
        elif format == 'actionable':
            return self._generate_actionable_report(analysis)
        else:
            return self._generate_detailed_report(analysis)
    
    def _generate_summary_report(self, analysis: Dict) -> str:
        """Generate a summary report."""
        report = f"""
Error Analysis Summary
====================
Student ID: {analysis['student_id']}
Date: {analysis['analysis_date']}

Total Errors: {analysis['total_errors']}
Severity: {analysis['severity_assessment']['overall'].upper()}

Top Error Types:
"""
        for error_type, count in sorted(analysis['error_types'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
            report += f"  - {error_type}: {count} occurrences\n"
        
        if analysis['misconceptions']:
            report += f"\nKey Misconceptions Detected: {len(analysis['misconceptions'])}\n"
            for misc in analysis['misconceptions'][:2]:
                report += f"  - {misc['type']}: {misc['description']}\n"
        
        if analysis['severity_assessment']['immediate_intervention_needed']:
            report += "\n⚠️  IMMEDIATE INTERVENTION RECOMMENDED\n"
        
        return report
    
    def _generate_actionable_report(self, analysis: Dict) -> str:
        """Generate an actionable report with specific next steps."""
        report = f"""
Actionable Error Analysis Report
===============================
Student ID: {analysis['student_id']}

IMMEDIATE ACTIONS REQUIRED:
"""
        
        # Priority 1: Address critical misconceptions
        critical_misconceptions = [m for m in analysis['misconceptions'] 
                                 if m.get('severity') in ['high', 'critical']]
        
        if critical_misconceptions:
            report += "\n1. Address Critical Misconceptions:\n"
            for misc in critical_misconceptions:
                report += f"   - {misc['type']}: {misc['remediation']['immediate_action']}\n"
        
        # Priority 2: Break persistent patterns
        persistent_patterns = [p for p in analysis['error_patterns'] 
                             if p['consistency'] > 0.7]
        
        if persistent_patterns:
            report += "\n2. Break Persistent Error Patterns:\n"
            for pattern in persistent_patterns[:3]:
                report += f"   - {pattern['type']} in {pattern['skill_area']}: "
                report += f"Use targeted exercises focusing on {pattern['triggers']}\n"
        
        # Priority 3: General recommendations
        report += "\n3. General Recommendations:\n"
        for rec in analysis['recommendations'][:5]:
            report += f"   - {rec['action']}: {rec['description']}\n"
        
        return report
    
    def _generate_detailed_report(self, analysis: Dict) -> str:
        """Generate a detailed report with all findings."""
        report = f"""
Comprehensive Error Analysis Report
==================================
Student ID: {analysis['student_id']}
Analysis Date: {analysis['analysis_date']}

Executive Summary
----------------
Total Errors Analyzed: {analysis['total_errors']}
Overall Severity: {analysis['severity_assessment']['overall'].upper()}
Immediate Intervention Needed: {'YES' if analysis['severity_assessment']['immediate_intervention_needed'] else 'NO'}

Error Type Distribution
----------------------
"""
        for error_type, count in sorted(analysis['error_types'].items(), 
                                      key=lambda x: x[1], reverse=True):
            percentage = (count / analysis['total_errors']) * 100
            report += f"  {error_type}: {count} ({percentage:.1f}%)\n"
        
        report += "\nError Patterns Identified\n"
        report += "------------------------\n"
        for i, pattern in enumerate(analysis['error_patterns'][:5], 1):
            report += f"\nPattern {i}:\n"
            report += f"  Type: {pattern['type']}\n"
            report += f"  Skill Area: {pattern['skill_area']}\n"
            report += f"  Frequency: {pattern['frequency']}\n"
            report += f"  Consistency: {pattern['consistency']:.2f}\n"
            report += f"  Triggers: {', '.join(pattern['triggers'])}\n"
        
        report += "\nMisconceptions Detected\n"
        report += "----------------------\n"
        for misc in analysis['misconceptions']:
            report += f"\n{misc['type']}:\n"
            report += f"  Description: {misc['description']}\n"
            report += f"  Frequency: {misc['frequency']}\n"
            report += f"  Severity: {misc.get('severity', 'moderate')}\n"
            report += f"  Remediation: {misc['remediation']['strategy']}\n"
        
        report += "\nProgress Indicators\n"
        report += "------------------\n"
        indicators = analysis['progress_indicators']
        report += f"  Self-Correction Rate: {indicators['self_correction_rate']:.1%}\n"
        report += f"  Error Clustering: {indicators['error_clustering_coefficient']:.2f}\n"
        
        report += "\nRecommended Interventions\n"
        report += "------------------------\n"
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"\n{i}. {rec['action']}\n"
            report += f"   Priority: {rec['priority']}\n"
            report += f"   Description: {rec['description']}\n"
            report += f"   Expected Impact: {rec.get('expected_impact', 'Moderate')}\n"
            report += f"   Timeline: {rec.get('timeline', 'Immediate')}\n"
        
        return report


class ErrorClassifier:
    """Classify errors into specific types."""
    
    def __init__(self):
        """Initialize error classifier with predefined error types."""
        self.error_types = {
            'calculation_error': self._check_calculation_error,
            'conceptual_error': self._check_conceptual_error,
            'procedural_error': self._check_procedural_error,
            'careless_error': self._check_careless_error,
            'place_value_error': self._check_place_value_error,
            'operation_confusion': self._check_operation_confusion,
            'pattern_recognition_error': self._check_pattern_error,
            'memory_recall_error': self._check_memory_error
        }
    
    def classify_error(self, error_data: Dict) -> Dict:
        """
        Classify a single error into categories.
        
        Returns:
        --------
        dict
            Error classification with primary and secondary types
        """
        classification = {
            'primary_type': 'unclassified',
            'secondary_types': [],
            'confidence': 0.0,
            'skill_area': error_data.get('skill_area', 'general'),
            'context': error_data.get('context', {})
        }
        
        # Check each error type
        scores = {}
        for error_type, check_func in self.error_types.items():
            score = check_func(error_data)
            if score > 0:
                scores[error_type] = score
        
        # Determine primary type
        if scores:
            primary = max(scores, key=scores.get)
            classification['primary_type'] = primary
            classification['confidence'] = scores[primary]
            
            # Add secondary types
            for error_type, score in scores.items():
                if error_type != primary and score > 0.3:
                    classification['secondary_types'].append(error_type)
        
        return classification
    
    def _check_calculation_error(self, error_data: Dict) -> float:
        """Check if error is a calculation mistake."""
        question = error_data.get('question', '')
        student_answer = str(error_data.get('student_answer', ''))
        correct_answer = str(error_data.get('correct_answer', ''))
        
        # Check if it's a numeric problem
        if not (self._is_numeric(student_answer) and self._is_numeric(correct_answer)):
            return 0.0
        
        # Check if answer is close but wrong (suggesting calculation error)
        try:
            student_val = float(student_answer)
            correct_val = float(correct_answer)
            
            # Within 20% suggests calculation error
            if abs(student_val - correct_val) / max(abs(correct_val), 1) < 0.2:
                return 0.8
            
            # Check for off-by-one errors
            if abs(student_val - correct_val) == 1:
                return 0.7
            
            # Check for sign errors
            if student_val == -correct_val:
                return 0.9
                
        except:
            pass
        
        return 0.0
    
    def _check_conceptual_error(self, error_data: Dict) -> float:
        """Check if error shows conceptual misunderstanding."""
        # Conceptual errors often show consistent wrong approach
        context = error_data.get('context', {})
        
        # If the same type of error appears multiple times
        if context.get('repeated_error', False):
            return 0.8
        
        # If error shows fundamental misunderstanding
        if context.get('shows_misconception', False):
            return 0.9
        
        # Check for wrong operation usage
        question = error_data.get('question', '')
        if self._wrong_operation_used(question, error_data):
            return 0.7
        
        return 0.0
    
    def _check_procedural_error(self, error_data: Dict) -> float:
        """Check if error is in following procedures."""
        context = error_data.get('context', {})
        
        # Procedural errors often occur in multi-step problems
        if context.get('problem_steps', 1) > 2:
            # Check if early steps were correct but later failed
            if context.get('partial_credit', 0) > 0.3:
                return 0.7
        
        # Check for order of operations errors
        question = error_data.get('question', '')
        if any(op in question for op in ['(', ')', '*', '/', '+', '-']):
            if context.get('order_error', False):
                return 0.8
        
        return 0.0
    
    def _check_careless_error(self, error_data: Dict) -> float:
        """Check if error appears to be careless mistake."""
        context = error_data.get('context', {})
        
        # Careless errors often happen under time pressure
        if context.get('time_pressure', 0) > 0.7:
            return 0.5
        
        # Or when problem is easy but student got it wrong
        if context.get('difficulty', 1) < 0.3:
            student_answer = str(error_data.get('student_answer', ''))
            correct_answer = str(error_data.get('correct_answer', ''))
            
            # Check for transcription errors
            if self._is_transcription_error(student_answer, correct_answer):
                return 0.8
        
        # If student usually gets similar problems right
        if context.get('usually_correct', False):
            return 0.6
        
        return 0.0
    
    def _check_place_value_error(self, error_data: Dict) -> float:
        """Check for place value understanding errors."""
        student_answer = str(error_data.get('student_answer', ''))
        correct_answer = str(error_data.get('correct_answer', ''))
        
        if not (self._is_numeric(student_answer) and self._is_numeric(correct_answer)):
            return 0.0
        
        try:
            student_val = float(student_answer)
            correct_val = float(correct_answer)
            
            # Check for power of 10 errors
            ratio = student_val / correct_val if correct_val != 0 else 0
            if ratio in [0.1, 0.01, 0.001, 10, 100, 1000]:
                return 0.9
            
            # Check for digit reversal (e.g., 32 instead of 23)
            if len(student_answer) == len(correct_answer) == 2:
                if student_answer[0] == correct_answer[1] and student_answer[1] == correct_answer[0]:
                    return 0.8
                    
        except:
            pass
        
        return 0.0
    
    def _check_operation_confusion(self, error_data: Dict) -> float:
        """Check if student confused operations."""
        question = error_data.get('question', '')
        student_answer = str(error_data.get('student_answer', ''))
        correct_answer = str(error_data.get('correct_answer', ''))
        
        # Look for operation keywords
        operations = {
            'add': ['+', 'plus', 'sum', 'total', 'altogether'],
            'subtract': ['-', 'minus', 'difference', 'less', 'fewer'],
            'multiply': ['×', '*', 'times', 'product', 'each'],
            'divide': ['÷', '/', 'split', 'share', 'per']
        }
        
        # Detect intended operation
        intended_op = None
        for op, keywords in operations.items():
            if any(keyword in question.lower() for keyword in keywords):
                intended_op = op
                break
        
        if not intended_op:
            return 0.0
        
        # Check if answer matches different operation
        # This is simplified - real implementation would be more sophisticated
        context = error_data.get('context', {})
        if context.get('wrong_operation', False):
            return 0.8
        
        return 0.0
    
    def _check_pattern_error(self, error_data: Dict) -> float:
        """Check for pattern recognition errors."""
        question = error_data.get('question', '')
        
        # Pattern problems often have keywords
        pattern_keywords = ['pattern', 'sequence', 'next', 'rule', 'continue']
        
        if any(keyword in question.lower() for keyword in pattern_keywords):
            context = error_data.get('context', {})
            if context.get('pattern_type', '') in ['numeric', 'geometric', 'algebraic']:
                return 0.7
        
        return 0.0
    
    def _check_memory_error(self, error_data: Dict) -> float:
        """Check for memory/recall errors."""
        context = error_data.get('context', {})
        
        # Memory errors often occur with math facts
        if context.get('problem_type') == 'math_fact':
            # Check if it's a commonly confused fact
            question = error_data.get('question', '')
            if any(fact in question for fact in ['7×8', '8×7', '6×9', '9×6', '7×6', '6×7']):
                return 0.6
        
        # Or when working memory is overloaded
        if context.get('problem_steps', 1) > 3:
            return 0.5
        
        return 0.0
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a number."""
        try:
            float(value)
            return True
        except:
            return False
    
    def _is_transcription_error(self, answer1: str, answer2: str) -> bool:
        """Check if two answers differ by simple transcription error."""
        if len(answer1) != len(answer2):
            return False
        
        # Count character differences
        differences = sum(1 for a, b in zip(answer1, answer2) if a != b)
        
        # Single character difference suggests transcription error
        return differences == 1
    
    def _wrong_operation_used(self, question: str, error_data: Dict) -> bool:
        """Check if wrong mathematical operation was used."""
        # This is a simplified check - real implementation would be more sophisticated
        context = error_data.get('context', {})
        return context.get('wrong_operation', False)


class MisconceptionDatabase:
    """Database of common mathematical misconceptions."""
    
    def __init__(self):
        """Initialize with common misconceptions."""
        self.misconceptions = {
            'always_add': {
                'description': 'Student adds numbers regardless of operation required',
                'indicators': ['adds when should subtract', 'ignores operation signs'],
                'severity': 'high',
                'remediation': {
                    'strategy': 'Operation discrimination training',
                    'immediate_action': 'Use visual cues to distinguish operations',
                    'exercises': ['operation_sorting', 'keyword_mapping']
                }
            },
            'larger_means_subtract': {
                'description': 'Student always subtracts smaller from larger',
                'indicators': ['negative answers become positive', 'order confusion'],
                'severity': 'high',
                'remediation': {
                    'strategy': 'Number line work with directed numbers',
                    'immediate_action': 'Introduce negative numbers conceptually',
                    'exercises': ['number_line_jumps', 'temperature_problems']
                }
            },
            'multiplication_is_repeated_addition': {
                'description': 'Student only understands multiplication as repeated addition',
                'indicators': ['struggles with decimals', 'cannot multiply fractions'],
                'severity': 'moderate',
                'remediation': {
                    'strategy': 'Area model and scaling concepts',
                    'immediate_action': 'Introduce multiplication as scaling',
                    'exercises': ['area_models', 'scaling_activities']
                }
            },
            'place_value_face_value': {
                'description': 'Student confuses place value with face value',
                'indicators': ['writes 23 as 203', 'adds digits incorrectly'],
                'severity': 'critical',
                'remediation': {
                    'strategy': 'Extensive place value work with manipulatives',
                    'immediate_action': 'Use base-10 blocks daily',
                    'exercises': ['place_value_charts', 'expanded_form']
                }
            },
            'fraction_means_two_numbers': {
                'description': 'Student treats fraction as two separate numbers',
                'indicators': ['adds numerators and denominators separately', 'no concept of parts'],
                'severity': 'high',
                'remediation': {
                    'strategy': 'Visual fraction models',
                    'immediate_action': 'Use pizza/pie models extensively',
                    'exercises': ['fraction_circles', 'equivalent_fractions']
                }
            }
        }
    
    def check_misconception(self, error_type: str, student_answer: str, 
                          correct_answer: str, question: str) -> Optional[Dict]:
        """
        Check if error indicates a known misconception.
        
        Returns:
        --------
        dict or None
            Misconception details if found
        """
        # Check for always adding
        if '-' in question or 'subtract' in question.lower():
            if self._seems_like_addition(student_answer, correct_answer, question):
                return {
                    'type': 'always_add',
                    **self.misconceptions['always_add']
                }
        
        # Check for place value confusion
        if error_type == 'place_value_error':
            return {
                'type': 'place_value_face_value',
                **self.misconceptions['place_value_face_value']
            }
        
        # Check for fraction misconceptions
        if '/' in question or 'fraction' in question.lower():
            if self._fraction_added_wrong(student_answer, question):
                return {
                    'type': 'fraction_means_two_numbers',
                    **self.misconceptions['fraction_means_two_numbers']
                }
        
        return None
    
    def _seems_like_addition(self, student_answer: str, correct_answer: str, 
                           question: str) -> bool:
        """Check if student added when should have done different operation."""
        # Extract numbers from question
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            try:
                # Check if student answer equals sum
                num_sum = sum(int(n) for n in numbers[:2])
                return str(num_sum) == student_answer and student_answer != correct_answer
            except:
                pass
        return False
    
    def _fraction_added_wrong(self, student_answer: str, question: str) -> bool:
        """Check if student added fractions incorrectly."""
        # Look for pattern like "1/2 + 1/3 = 2/5"
        fractions = re.findall(r'(\d+)/(\d+)', question)
        if len(fractions) >= 2:
            # Check if student added numerators and denominators separately
            try:
                num_sum = sum(int(f[0]) for f in fractions)
                den_sum = sum(int(f[1]) for f in fractions)
                wrong_answer = f"{num_sum}/{den_sum}"
                return wrong_answer == student_answer
            except:
                pass
        return False


class RemediationEngine:
    """Generate targeted remediation strategies based on error analysis."""
    
    def generate_recommendations(self, error_types: Dict[str, int], 
                               patterns: List[Dict],
                               misconceptions: List[Dict], 
                               severity: Dict) -> List[Dict]:
        """
        Generate prioritized recommendations based on error analysis.
        
        Returns:
        --------
        list
            List of recommendation objects
        """
        recommendations = []
        
        # Priority 1: Address critical misconceptions
        for misc in misconceptions:
            if misc.get('severity') in ['critical', 'high']:
                rec = {
                    'action': f"Address {misc['type']} misconception",
                    'priority': 'critical' if misc['severity'] == 'critical' else 'high',
                    'description': misc['remediation']['immediate_action'],
                    'exercises': misc['remediation']['exercises'],
                    'timeline': 'immediate',
                    'expected_impact': 'high',
                    'resources': self._get_resources_for_misconception(misc['type'])
                }
                recommendations.append(rec)
        
        # Priority 2: Break persistent patterns
        for pattern in patterns:
            if pattern['consistency'] > 0.7 and pattern['frequency'] > 5:
                rec = {
                    'action': f"Break {pattern['type']} pattern in {pattern['skill_area']}",
                    'priority': 'high',
                    'description': self._get_pattern_intervention(pattern),
                    'exercises': self._get_pattern_exercises(pattern),
                    'timeline': '1-2 weeks',
                    'expected_impact': 'moderate',
                    'monitoring': 'daily'
                }
                recommendations.append(rec)
        
        # Priority 3: General skill building
        for error_type, count in error_types.items():
            if count > 3:
                rec = {
                    'action': f"Strengthen {error_type} skills",
                    'priority': 'moderate',
                    'description': self._get_skill_building_strategy(error_type),
                    'exercises': self._get_skill_exercises(error_type),
                    'timeline': '2-4 weeks',
                    'expected_impact': 'moderate',
                    'assessment': 'weekly'
                }
                recommendations.append(rec)
        
        # Priority 4: Preventive measures
        if severity['overall'] in ['moderate', 'high', 'critical']:
            rec = {
                'action': 'Implement preventive strategies',
                'priority': 'moderate',
                'description': 'Regular review sessions and confidence building',
                'exercises': ['review_sessions', 'confidence_builders', 'success_experiences'],
                'timeline': 'ongoing',
                'expected_impact': 'long-term'
            }
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'moderate': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _get_resources_for_misconception(self, misconception_type: str) -> List[str]:
        """Get learning resources for specific misconception."""
        resources_map = {
            'always_add': [
                'Operation sorting cards',
                'Word problem keyword chart',
                'Interactive operation games'
            ],
            'place_value_face_value': [
                'Base-10 blocks',
                'Place value charts',
                'Digital place value manipulatives'
            ],
            'fraction_means_two_numbers': [
                'Fraction circles',
                'Fraction bars',
                'Virtual fraction tools'
            ]
        }
        
        return resources_map.get(misconception_type, ['General math manipulatives'])
    
    def _get_pattern_intervention(self, pattern: Dict) -> str:
        """Get intervention strategy for error pattern."""
        interventions = {
            'calculation_error': 'Focus on accuracy over speed, use checking strategies',
            'conceptual_error': 'Reteach fundamental concepts with concrete examples',
            'procedural_error': 'Practice step-by-step procedures with scaffolding',
            'careless_error': 'Implement self-checking routines and mindfulness techniques'
        }
        
        return interventions.get(pattern['type'], 
                                'Targeted practice with immediate feedback')
    
    def _get_pattern_exercises(self, pattern: Dict) -> List[str]:
        """Get exercises to address specific error pattern."""
        exercises_map = {
            'calculation_error': [
                'estimation_before_calculation',
                'check_by_different_method',
                'mental_math_strategies'
            ],
            'conceptual_error': [
                'concept_mapping',
                'real_world_applications',
                'peer_teaching'
            ],
            'procedural_error': [
                'step_by_step_guides',
                'worked_examples',
                'error_finding_exercises'
            ]
        }
        
        return exercises_map.get(pattern['type'], ['targeted_practice'])
    
    def _get_skill_building_strategy(self, error_type: str) -> str:
        """Get strategy for building specific skills."""
        strategies = {
            'memory_recall_error': 'Use spaced repetition and mnemonic devices',
            'pattern_recognition_error': 'Practice with varied pattern types',
            'place_value_error': 'Intensive place value instruction with manipulatives',
            'operation_confusion': 'Operation discrimination exercises'
        }
        
        return strategies.get(error_type, 'Targeted skill practice with feedback')
    
    def _get_skill_exercises(self, error_type: str) -> List[str]:
        """Get exercises for specific skill building."""
        exercises = {
            'memory_recall_error': ['flashcards', 'timed_drills', 'memory_games'],
            'pattern_recognition_error': ['pattern_completion', 'pattern_creation', 'pattern_analysis'],
            'place_value_error': ['place_value_sorting', 'expanded_form_practice', 'base_10_activities'],
            'operation_confusion': ['operation_sorting', 'word_problem_analysis', 'operation_matching']
        }
        
        return exercises.get(error_type, ['skill_practice', 'application_problems'])


# Example usage and testing
if __name__ == "__main__":
    # Initialize the error pattern analyzer
    analyzer = ErrorPatternAnalyzer()
    
    # Sample error data
    sample_errors = [
        {
            'question': 'What is 25 - 13?',
            'student_answer': '38',
            'correct_answer': '12',
            'skill_area': 'subtraction',
            'context': {
                'difficulty': 0.3,
                'time_pressure': 0.2,
                'problem_type': 'basic_arithmetic'
            }
        },
        {
            'question': 'What is 45 - 28?',
            'student_answer': '73',
            'correct_answer': '17',
            'skill_area': 'subtraction',
            'context': {
                'difficulty': 0.4,
                'time_pressure': 0.3,
                'problem_type': 'basic_arithmetic'
            }
        },
        {
            'question': 'What is 234 + 567?',
            'student_answer': '791',
            'correct_answer': '801',
            'skill_area': 'addition',
            'context': {
                'difficulty': 0.5,
                'time_pressure': 0.5,
                'problem_type': 'multi_digit'
            }
        },
        {
            'question': 'What is 1/2 + 1/3?',
            'student_answer': '2/5',
            'correct_answer': '5/6',
            'skill_area': 'fractions',
            'context': {
                'difficulty': 0.7,
                'time_pressure': 0.4,
                'problem_type': 'fraction_addition'
            }
        }
    ]
    
    # Analyze errors
    analysis = analyzer.analyze_student_errors('student_123', sample_errors)
    
    # Generate reports
    print("=== SUMMARY REPORT ===")
    print(analyzer.generate_error_report(analysis, format='summary'))
    
    print("\n=== ACTIONABLE REPORT ===")
    print(analyzer.generate_error_report(analysis, format='actionable'))
    
    # Show key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Total errors analyzed: {analysis['total_errors']}")
    print(f"Severity: {analysis['severity_assessment']['overall']}")
    print(f"Number of patterns found: {len(analysis['error_patterns'])}")
    print(f"Number of misconceptions: {len(analysis['misconceptions'])}")
    print(f"Recommendations: {len(analysis['recommendations'])}")