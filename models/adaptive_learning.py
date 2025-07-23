"""
Adaptive Learning System for MathemAI
This module provides personalized learning experiences by adapting to individual student
needs, learning styles, and progress patterns.
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Tuple, Optional


class AdaptiveLearningSystem:
    """
    Adaptive learning system that personalizes educational content based on student performance,
    learning style, and progress tracking.
    """
    
    def __init__(self):
        """Initialize the adaptive learning system."""
        self.student_profiles = {}
        self.exercise_bank = ExerciseBank()
        self.intervention_strategies = InterventionStrategies()
        self.progress_analyzer = ProgressAnalyzer()
        
    def create_student_profile(self, student_id: str, assessment_results: Dict) -> Dict:
        """
        Create a comprehensive student profile based on initial assessment.
        
        Parameters:
        -----------
        student_id : str
            Unique identifier for the student
        assessment_results : dict
            Initial assessment scores and data
            
        Returns:
        --------
        dict
            Complete student profile
        """
        profile = {
            'id': student_id,
            'created_at': datetime.now().isoformat(),
            'strengths': [],
            'weaknesses': [],
            'learning_style': self._determine_learning_style(assessment_results),
            'pace_preference': self._determine_pace(assessment_results),
            'current_levels': {},
            'skill_mastery': {},
            'progress_history': deque(maxlen=100),
            'intervention_history': [],
            'performance_trends': {},
            'engagement_metrics': {
                'average_session_duration': 0,
                'exercises_completed': 0,
                'streak_days': 0,
                'last_active': datetime.now().isoformat()
            },
            'preferences': {
                'difficulty_adjustment_rate': 0.1,
                'exercise_variety': 'high',
                'feedback_frequency': 'immediate'
            }
        }
        
        # Analyze strengths and weaknesses
        for skill, score in assessment_results.items():
            if isinstance(score, (int, float)):
                if score > 0.75:
                    profile['strengths'].append(skill)
                elif score < 0.4:
                    profile['weaknesses'].append(skill)
                
                # Set initial skill levels
                profile['current_levels'][skill] = self._determine_skill_level(score)
                profile['skill_mastery'][skill] = score
        
        # Initialize performance trends
        for skill in profile['current_levels']:
            profile['performance_trends'][skill] = {
                'direction': 'stable',
                'confidence': 0.5,
                'last_updated': datetime.now().isoformat()
            }
        
        self.student_profiles[student_id] = profile
        return profile
    
    def _determine_learning_style(self, assessment_results: Dict) -> str:
        """
        Determine student's preferred learning style based on assessment performance.
        
        Parameters:
        -----------
        assessment_results : dict
            Assessment scores
            
        Returns:
        --------
        str
            Learning style (visual, auditory, kinesthetic, reading_writing)
        """
        # Calculate learning style indicators
        visual_score = assessment_results.get('visual_spatial_score', 0)
        verbal_score = assessment_results.get('word_problem_solving', 0)
        kinesthetic_score = assessment_results.get('calculation_fluency', 0)
        reading_score = (assessment_results.get('number_recognition', 0) + 
                        assessment_results.get('arithmetic_facts_recall', 0)) / 2
        
        # Account for attention and working memory
        attention = assessment_results.get('attention_score', 0.5)
        memory = assessment_results.get('working_memory_score', 0.5)
        
        # Adjust scores based on cognitive factors
        visual_score *= (1 + attention * 0.2)
        kinesthetic_score *= (1 + (1 - attention) * 0.2)  # Kinesthetic learners may have lower attention
        reading_score *= (1 + memory * 0.2)
        
        styles = {
            'visual': visual_score,
            'auditory': verbal_score,
            'kinesthetic': kinesthetic_score,
            'reading_writing': reading_score
        }
        
        # Return primary learning style
        return max(styles, key=styles.get)
    
    def _determine_pace(self, assessment_results: Dict) -> str:
        """
        Determine student's learning pace preference.
        
        Returns:
        --------
        str
            Pace preference (slow, moderate, fast)
        """
        # Factors affecting learning pace
        fluency = assessment_results.get('calculation_fluency', 0.5)
        accuracy = assessment_results.get('calculation_accuracy', 0.5)
        memory = assessment_results.get('working_memory_score', 0.5)
        
        pace_score = (fluency * 0.4 + accuracy * 0.3 + memory * 0.3)
        
        if pace_score < 0.4:
            return 'slow'
        elif pace_score < 0.7:
            return 'moderate'
        else:
            return 'fast'
    
    def _determine_skill_level(self, score: float) -> str:
        """
        Determine skill level based on score.
        
        Parameters:
        -----------
        score : float
            Skill score (0-1)
            
        Returns:
        --------
        str
            Skill level
        """
        if score < 0.3:
            return 'beginner'
        elif score < 0.5:
            return 'elementary'
        elif score < 0.7:
            return 'intermediate'
        elif score < 0.85:
            return 'advanced'
        else:
            return 'expert'
    
    def generate_personalized_lesson(self, student_id: str, skill_area: str, 
                                   duration_minutes: int = 30) -> Dict:
        """
        Generate a personalized lesson plan for a specific skill area.
        
        Parameters:
        -----------
        student_id : str
            Student identifier
        skill_area : str
            Skill area to focus on
        duration_minutes : int
            Lesson duration in minutes
            
        Returns:
        --------
        dict
            Personalized lesson plan
        """
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Student profile not found for ID: {student_id}")
        
        current_level = profile['current_levels'].get(skill_area, 'beginner')
        learning_style = profile['learning_style']
        pace = profile['pace_preference']
        
        # Calculate exercise distribution
        exercise_count = self._calculate_exercise_count(duration_minutes, pace)
        
        lesson = {
            'student_id': student_id,
            'skill_area': skill_area,
            'duration_minutes': duration_minutes,
            'created_at': datetime.now().isoformat(),
            'exercises': [],
            'learning_objectives': self._generate_learning_objectives(skill_area, current_level),
            'resources': []
        }
        
        # Generate exercises based on learning style
        for i in range(exercise_count):
            difficulty = self._calculate_exercise_difficulty(profile, skill_area, i, exercise_count)
            exercise = self.exercise_bank.get_exercise(
                skill_area=skill_area,
                difficulty=difficulty,
                learning_style=learning_style,
                exercise_number=i+1
            )
            lesson['exercises'].append(exercise)
        
        # Add resources based on learning style
        lesson['resources'] = self._get_learning_resources(skill_area, learning_style, current_level)
        
        # Add adaptive elements
        lesson['adaptive_rules'] = {
            'success_threshold': 0.8,
            'struggle_threshold': 0.5,
            'adjustment_actions': {
                'success': 'increase_difficulty',
                'struggle': 'provide_hint_or_decrease_difficulty',
                'timeout': 'provide_worked_example'
            }
        }
        
        return lesson
    
    def _calculate_exercise_count(self, duration_minutes: int, pace: str) -> int:
        """Calculate number of exercises based on duration and pace."""
        base_exercises_per_minute = {
            'slow': 0.5,
            'moderate': 0.75,
            'fast': 1.0
        }
        
        exercises_per_minute = base_exercises_per_minute.get(pace, 0.75)
        return max(3, int(duration_minutes * exercises_per_minute))
    
    def _calculate_exercise_difficulty(self, profile: Dict, skill_area: str, 
                                     exercise_index: int, total_exercises: int) -> float:
        """
        Calculate appropriate difficulty for an exercise.
        
        Returns:
        --------
        float
            Difficulty level (0-1)
        """
        base_level = profile['skill_mastery'].get(skill_area, 0.5)
        
        # Progressive difficulty within lesson
        progression_factor = exercise_index / max(1, total_exercises - 1)
        
        # Adjust based on recent performance
        recent_performance = self._get_recent_performance(profile, skill_area)
        
        if recent_performance > 0.8:
            # Performing well, increase difficulty
            difficulty_boost = 0.1
        elif recent_performance < 0.5:
            # Struggling, decrease difficulty
            difficulty_boost = -0.1
        else:
            difficulty_boost = 0
        
        # Calculate final difficulty
        difficulty = base_level + (progression_factor * 0.2) + difficulty_boost
        
        # Clamp between 0 and 1
        return max(0.1, min(0.95, difficulty))
    
    def _get_recent_performance(self, profile: Dict, skill_area: str) -> float:
        """Get recent performance average for a skill area."""
        recent_entries = [
            entry for entry in profile['progress_history']
            if entry.get('skill_area') == skill_area
        ][-5:]  # Last 5 entries
        
        if not recent_entries:
            return profile['skill_mastery'].get(skill_area, 0.5)
        
        performances = [entry.get('performance', 0.5) for entry in recent_entries]
        return np.mean(performances)
    
    def _generate_learning_objectives(self, skill_area: str, level: str) -> List[str]:
        """Generate specific learning objectives for the lesson."""
        objectives_map = {
            'number_recognition': {
                'beginner': ['Identify single-digit numbers', 'Match numbers to quantities'],
                'elementary': ['Recognize two-digit numbers', 'Understand place value basics'],
                'intermediate': ['Work with three-digit numbers', 'Compare and order numbers'],
                'advanced': ['Handle large numbers', 'Understand number patterns'],
                'expert': ['Master complex number relationships', 'Apply number theory concepts']
            },
            'calculation_accuracy': {
                'beginner': ['Perform simple addition/subtraction', 'Use counting strategies'],
                'elementary': ['Add/subtract with regrouping', 'Introduce multiplication'],
                'intermediate': ['Multi-digit operations', 'Basic division'],
                'advanced': ['Complex calculations', 'Mental math strategies'],
                'expert': ['Advanced problem solving', 'Efficient calculation methods']
            }
        }
        
        # Default objectives if skill area not mapped
        default_objectives = {
            'beginner': ['Build foundational understanding', 'Practice basic concepts'],
            'elementary': ['Strengthen core skills', 'Increase accuracy'],
            'intermediate': ['Develop fluency', 'Apply concepts to problems'],
            'advanced': ['Master complex applications', 'Develop problem-solving strategies'],
            'expert': ['Achieve automaticity', 'Transfer skills to new contexts']
        }
        
        return objectives_map.get(skill_area, default_objectives).get(level, default_objectives['intermediate'])
    
    def _get_learning_resources(self, skill_area: str, learning_style: str, level: str) -> List[Dict]:
        """Get appropriate learning resources based on style and level."""
        resources = []
        
        # Visual learners
        if learning_style == 'visual':
            resources.extend([
                {'type': 'video', 'description': f'Visual tutorial on {skill_area}'},
                {'type': 'infographic', 'description': f'{skill_area} concept map'},
                {'type': 'interactive_diagram', 'description': 'Interactive visual aids'}
            ])
        
        # Auditory learners
        elif learning_style == 'auditory':
            resources.extend([
                {'type': 'audio_explanation', 'description': f'Verbal explanation of {skill_area}'},
                {'type': 'mnemonic_songs', 'description': 'Memory songs for key concepts'},
                {'type': 'discussion_prompts', 'description': 'Questions for verbal practice'}
            ])
        
        # Kinesthetic learners
        elif learning_style == 'kinesthetic':
            resources.extend([
                {'type': 'manipulatives', 'description': 'Physical objects for hands-on learning'},
                {'type': 'movement_activities', 'description': 'Learning through movement'},
                {'type': 'interactive_games', 'description': 'Active learning games'}
            ])
        
        # Reading/Writing learners
        else:
            resources.extend([
                {'type': 'worksheet', 'description': f'Practice problems for {skill_area}'},
                {'type': 'written_explanation', 'description': 'Step-by-step written guide'},
                {'type': 'note_templates', 'description': 'Structured note-taking templates'}
            ])
        
        return resources
    
    def update_student_progress(self, student_id: str, exercise_results: List[Dict]) -> Dict:
        """
        Update student progress based on completed exercises.
        
        Parameters:
        -----------
        student_id : str
            Student identifier
        exercise_results : list
            List of exercise results
            
        Returns:
        --------
        dict
            Updated progress summary
        """
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Student profile not found for ID: {student_id}")
        
        progress_summary = {
            'exercises_completed': len(exercise_results),
            'average_performance': 0,
            'skill_updates': {},
            'achievements': [],
            'recommendations': []
        }
        
        # Process each exercise result
        skill_performances = defaultdict(list)
        
        for result in exercise_results:
            skill_area = result.get('skill_area')
            performance = result.get('performance', 0)
            time_taken = result.get('time_taken', 0)
            
            skill_performances[skill_area].append(performance)
            
            # Add to progress history
            progress_entry = {
                'timestamp': datetime.now().isoformat(),
                'skill_area': skill_area,
                'performance': performance,
                'time_taken': time_taken,
                'difficulty': result.get('difficulty', 0.5)
            }
            profile['progress_history'].append(progress_entry)
        
        # Update skill mastery and levels
        for skill_area, performances in skill_performances.items():
            avg_performance = np.mean(performances)
            
            # Update skill mastery with weighted average
            old_mastery = profile['skill_mastery'].get(skill_area, 0.5)
            new_mastery = old_mastery * 0.7 + avg_performance * 0.3
            profile['skill_mastery'][skill_area] = new_mastery
            
            # Update skill level if needed
            new_level = self._determine_skill_level(new_mastery)
            old_level = profile['current_levels'].get(skill_area, 'beginner')
            
            if new_level != old_level:
                profile['current_levels'][skill_area] = new_level
                progress_summary['achievements'].append({
                    'type': 'level_up',
                    'skill_area': skill_area,
                    'old_level': old_level,
                    'new_level': new_level
                })
            
            progress_summary['skill_updates'][skill_area] = {
                'old_mastery': old_mastery,
                'new_mastery': new_mastery,
                'performance': avg_performance,
                'trend': self._calculate_trend(profile, skill_area)
            }
        
        # Update engagement metrics
        profile['engagement_metrics']['exercises_completed'] += len(exercise_results)
        profile['engagement_metrics']['last_active'] = datetime.now().isoformat()
        
        # Calculate overall performance
        all_performances = [p for perfs in skill_performances.values() for p in perfs]
        progress_summary['average_performance'] = np.mean(all_performances) if all_performances else 0
        
        # Generate recommendations
        progress_summary['recommendations'] = self._generate_progress_recommendations(profile, skill_performances)
        
        return progress_summary
    
    def _calculate_trend(self, profile: Dict, skill_area: str) -> str:
        """Calculate performance trend for a skill area."""
        recent_entries = [
            entry for entry in profile['progress_history']
            if entry.get('skill_area') == skill_area
        ][-10:]  # Last 10 entries
        
        if len(recent_entries) < 3:
            return 'insufficient_data'
        
        performances = [entry.get('performance', 0.5) for entry in recent_entries]
        
        # Simple linear regression to find trend
        x = np.arange(len(performances))
        slope = np.polyfit(x, performances, 1)[0]
        
        if slope > 0.02:
            return 'improving'
        elif slope < -0.02:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_progress_recommendations(self, profile: Dict, 
                                         skill_performances: Dict[str, List[float]]) -> List[Dict]:
        """Generate recommendations based on progress."""
        recommendations = []
        
        # Check for struggling areas
        for skill_area, performances in skill_performances.items():
            avg_performance = np.mean(performances)
            
            if avg_performance < 0.5:
                recommendations.append({
                    'type': 'intervention',
                    'skill_area': skill_area,
                    'urgency': 'high',
                    'suggestion': f'Consider additional support for {skill_area}',
                    'strategies': self.intervention_strategies.get_strategies(skill_area, profile['learning_style'])
                })
            elif avg_performance > 0.85:
                recommendations.append({
                    'type': 'advancement',
                    'skill_area': skill_area,
                    'urgency': 'low',
                    'suggestion': f'Ready for more challenging {skill_area} content'
                })
        
        return recommendations
    
    def recommend_intervention(self, student_id: str) -> Dict:
        """
        Generate comprehensive intervention recommendations for a student.
        
        Parameters:
        -----------
        student_id : str
            Student identifier
            
        Returns:
        --------
        dict
            Intervention plan
        """
        profile = self.student_profiles.get(student_id)
        if not profile:
            raise ValueError(f"Student profile not found for ID: {student_id}")
        
        intervention_plan = {
            'student_id': student_id,
            'created_at': datetime.now().isoformat(),
            'priority_areas': [],
            'strategies': [],
            'timeline': {},
            'resources': [],
            'success_metrics': []
        }
        
        # Identify priority areas (weaknesses or declining performance)
        for skill_area in profile['weaknesses']:
            trend = profile['performance_trends'].get(skill_area, {}).get('direction', 'stable')
            mastery = profile['skill_mastery'].get(skill_area, 0.5)
            
            priority_score = (1 - mastery) * (1.5 if trend == 'declining' else 1.0)
            
            intervention_plan['priority_areas'].append({
                'skill_area': skill_area,
                'current_mastery': mastery,
                'trend': trend,
                'priority_score': priority_score
            })
        
        # Sort by priority
        intervention_plan['priority_areas'].sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Generate strategies for top priority areas
        for priority_area in intervention_plan['priority_areas'][:3]:  # Top 3 areas
            skill_area = priority_area['skill_area']
            strategies = self.intervention_strategies.get_comprehensive_plan(
                skill_area=skill_area,
                learning_style=profile['learning_style'],
                current_level=profile['current_levels'].get(skill_area, 'beginner')
            )
            intervention_plan['strategies'].extend(strategies)
        
        # Create timeline
        intervention_plan['timeline'] = self._create_intervention_timeline(intervention_plan['priority_areas'])
        
        # Add success metrics
        intervention_plan['success_metrics'] = [
            {'metric': 'skill_mastery_improvement', 'target': 0.2, 'timeframe': '4_weeks'},
            {'metric': 'exercise_completion_rate', 'target': 0.8, 'timeframe': 'weekly'},
            {'metric': 'performance_trend', 'target': 'improving', 'timeframe': '2_weeks'}
        ]
        
        return intervention_plan
    
    def _create_intervention_timeline(self, priority_areas: List[Dict]) -> Dict:
        """Create a timeline for intervention implementation."""
        timeline = {
            'week_1': [],
            'week_2': [],
            'week_3': [],
            'week_4': [],
            'ongoing': []
        }
        
        # Distribute interventions across weeks
        for i, area in enumerate(priority_areas[:3]):
            skill_area = area['skill_area']
            
            # Intensive focus in first two weeks
            timeline['week_1'].append({
                'skill_area': skill_area,
                'intensity': 'high',
                'sessions_per_week': 4
            })
            timeline['week_2'].append({
                'skill_area': skill_area,
                'intensity': 'high',
                'sessions_per_week': 3
            })
            
            # Maintenance in later weeks
            timeline['week_3'].append({
                'skill_area': skill_area,
                'intensity': 'moderate',
                'sessions_per_week': 2
            })
            timeline['week_4'].append({
                'skill_area': skill_area,
                'intensity': 'moderate',
                'sessions_per_week': 2
            })
        
        timeline['ongoing'] = [
            {'activity': 'progress_monitoring', 'frequency': 'daily'},
            {'activity': 'adaptation_review', 'frequency': 'weekly'}
        ]
        
        return timeline


class ExerciseBank:
    """Bank of exercises for different skill areas and difficulty levels."""
    
    def get_exercise(self, skill_area: str, difficulty: float, 
                    learning_style: str, exercise_number: int) -> Dict:
        """
        Get an appropriate exercise based on parameters.
        
        Returns:
        --------
        dict
            Exercise details
        """
        exercise = {
            'id': f"{skill_area}_{difficulty:.2f}_{exercise_number}",
            'skill_area': skill_area,
            'difficulty': difficulty,
            'learning_style': learning_style,
            'type': self._get_exercise_type(learning_style),
            'content': self._generate_exercise_content(skill_area, difficulty, learning_style),
            'hints': self._generate_hints(skill_area, difficulty),
            'solution': self._generate_solution(skill_area, difficulty),
            'estimated_time': self._estimate_time(difficulty)
        }
        
        return exercise
    
    def _get_exercise_type(self, learning_style: str) -> str:
        """Get exercise type based on learning style."""
        type_map = {
            'visual': 'visual_problem',
            'auditory': 'verbal_problem',
            'kinesthetic': 'interactive_problem',
            'reading_writing': 'written_problem'
        }
        return type_map.get(learning_style, 'standard_problem')
    
    def _generate_exercise_content(self, skill_area: str, difficulty: float, learning_style: str) -> Dict:
        """Generate exercise content based on parameters."""
        # Simplified content generation - in real implementation, this would be more sophisticated
        content = {
            'question': f"Practice {skill_area} at difficulty {difficulty:.2f}",
            'format': learning_style,
            'elements': []
        }
        
        if learning_style == 'visual':
            content['elements'] = ['diagram', 'color_coding', 'visual_representation']
        elif learning_style == 'kinesthetic':
            content['elements'] = ['drag_drop', 'manipulation', 'interactive_elements']
        
        return content
    
    def _generate_hints(self, skill_area: str, difficulty: float) -> List[str]:
        """Generate progressive hints for the exercise."""
        hint_count = max(1, int(3 * difficulty))
        hints = []
        
        for i in range(hint_count):
            hint_level = (i + 1) / hint_count
            hints.append(f"Hint {i+1}: Focus on {skill_area} concepts (specificity: {hint_level:.1f})")
        
        return hints
    
    def _generate_solution(self, skill_area: str, difficulty: float) -> Dict:
        """Generate solution with explanation."""
        return {
            'answer': 'Correct answer here',
            'explanation': f'Step-by-step explanation for {skill_area}',
            'common_mistakes': ['Mistake 1', 'Mistake 2'],
            'learning_points': ['Key concept 1', 'Key concept 2']
        }
    
    def _estimate_time(self, difficulty: float) -> int:
        """Estimate time needed for exercise in seconds."""
        base_time = 60  # 1 minute base
        return int(base_time * (1 + difficulty))


class InterventionStrategies:
    """Collection of intervention strategies for different learning needs."""
    
    def get_strategies(self, skill_area: str, learning_style: str) -> List[Dict]:
        """Get intervention strategies for a skill area and learning style."""
        strategies = []
        
        # Base strategies for skill area
        base_strategies = {
            'number_recognition': [
                {'name': 'Number line activities', 'description': 'Use visual number lines'},
                {'name': 'Counting games', 'description': 'Interactive counting exercises'},
                {'name': 'Number matching', 'description': 'Match numerals to quantities'}
            ],
            'calculation_accuracy': [
                {'name': 'Fact families', 'description': 'Practice related math facts'},
                {'name': 'Mental math strategies', 'description': 'Teach mental calculation tricks'},
                {'name': 'Error analysis', 'description': 'Review and learn from mistakes'}
            ]
        }
        
        # Get base strategies
        strategies.extend(base_strategies.get(skill_area, [
            {'name': 'Targeted practice', 'description': f'Focused exercises for {skill_area}'},
            {'name': 'Skill building', 'description': 'Progressive skill development'}
        ]))
        
        # Add learning style specific strategies
        if learning_style == 'visual':
            strategies.append({'name': 'Visual aids', 'description': 'Use charts, diagrams, and colors'})
        elif learning_style == 'kinesthetic':
            strategies.append({'name': 'Hands-on activities', 'description': 'Use manipulatives and movement'})
        
        return strategies
    
    def get_comprehensive_plan(self, skill_area: str, learning_style: str, current_level: str) -> List[Dict]:
        """Get a comprehensive intervention plan."""
        strategies = []
        
        # Immediate strategies
        strategies.append({
            'phase': 'immediate',
            'duration': '1_week',
            'strategies': self.get_strategies(skill_area, learning_style),
            'intensity': 'high'
        })
        
        # Short-term strategies
        strategies.append({
            'phase': 'short_term',
            'duration': '2_weeks',
            'strategies': [
                {'name': 'Scaffolded practice', 'description': 'Gradually increase difficulty'},
                {'name': 'Peer tutoring', 'description': 'Work with successful peers'}
            ],
            'intensity': 'moderate'
        })
        
        # Long-term strategies
        strategies.append({
            'phase': 'long_term',
            'duration': '4_weeks',
            'strategies': [
                {'name': 'Independent practice', 'description': 'Self-directed learning'},
                {'name': 'Real-world application', 'description': 'Apply skills to real situations'}
            ],
            'intensity': 'low'
        })
        
        return strategies


class ProgressAnalyzer:
    """Analyze student progress patterns and provide insights."""
    
    def analyze_progress(self, progress_history: List[Dict]) -> Dict:
        """
        Analyze progress history to identify patterns and insights.
        
        Returns:
        --------
        dict
            Progress analysis results
        """
        if not progress_history:
            return {'status': 'no_data'}
        
        analysis = {
            'overall_trend': 'stable',
            'consistency': 0,
            'strengths_emerging': [],
            'concerns': [],
            'recommendations': []
        }
        
        # Analyze by skill area
        skill_data = defaultdict(list)
        for entry in progress_history:
            skill_area = entry.get('skill_area')
            performance = entry.get('performance', 0)
            skill_data[skill_area].append(performance)
        
        # Calculate trends and consistency
        trends = {}
        for skill_area, performances in skill_data.items():
            if len(performances) >= 3:
                # Calculate trend
                x = np.arange(len(performances))
                slope = np.polyfit(x, performances, 1)[0]
                
                # Calculate consistency (inverse of standard deviation)
                consistency = 1 / (np.std(performances) + 0.1)
                
                trends[skill_area] = {
                    'slope': slope,
                    'consistency': consistency,
                    'average': np.mean(performances)
                }
        
        # Determine overall trend
        if trends:
            avg_slope = np.mean([t['slope'] for t in trends.values()])
            if avg_slope > 0.01:
                analysis['overall_trend'] = 'improving'
            elif avg_slope < -0.01:
                analysis['overall_trend'] = 'declining'
            
            analysis['consistency'] = np.mean([t['consistency'] for t in trends.values()])
        
        # Identify emerging strengths and concerns
        for skill_area, trend_data in trends.items():
            if trend_data['slope'] > 0.02 and trend_data['average'] > 0.7:
                analysis['strengths_emerging'].append(skill_area)
            elif trend_data['slope'] < -0.02 or trend_data['average'] < 0.4:
                analysis['concerns'].append({
                    'skill_area': skill_area,
                    'reason': 'declining_performance' if trend_data['slope'] < -0.02 else 'low_performance'
                })
        
        return analysis


# Example usage and testing
if __name__ == "__main__":
    # Initialize the adaptive learning system
    als = AdaptiveLearningSystem()
    
    # Create a sample student profile
    sample_assessment = {
        'number_recognition': 0.7,
        'number_comparison': 0.6,
        'counting_skills': 0.65,
        'calculation_accuracy': 0.4,
        'calculation_fluency': 0.35,
        'arithmetic_facts_recall': 0.5,
        'word_problem_solving': 0.3,
        'working_memory_score': 0.6,
        'visual_spatial_score': 0.7,
        'attention_score': 0.5
    }
    
    # Create student profile
    student_id = "student_001"
    profile = als.create_student_profile(student_id, sample_assessment)
    print("Student Profile Created:")
    print(f"  Learning Style: {profile['learning_style']}")
    print(f"  Pace: {profile['pace_preference']}")
    print(f"  Strengths: {profile['strengths']}")
    print(f"  Weaknesses: {profile['weaknesses']}")
    
    # Generate personalized lesson
    lesson = als.generate_personalized_lesson(student_id, 'calculation_accuracy', 30)
    print(f"\nPersonalized Lesson Generated:")
    print(f"  Skill Area: {lesson['skill_area']}")
    print(f"  Duration: {lesson['duration_minutes']} minutes")
    print(f"  Number of Exercises: {len(lesson['exercises'])}")
    print(f"  Learning Objectives: {lesson['learning_objectives']}")
    
    # Simulate exercise results
    exercise_results = [
        {'skill_area': 'calculation_accuracy', 'performance': 0.6, 'time_taken': 120},
        {'skill_area': 'calculation_accuracy', 'performance': 0.65, 'time_taken': 110},
        {'skill_area': 'calculation_accuracy', 'performance': 0.7, 'time_taken': 100}
    ]
    
    # Update progress
    progress_summary = als.update_student_progress(student_id, exercise_results)
    print(f"\nProgress Updated:")
    print(f"  Average Performance: {progress_summary['average_performance']:.2f}")
    print(f"  Skill Updates: {progress_summary['skill_updates']}")
    
    # Generate intervention recommendations
    intervention = als.recommend_intervention(student_id)
    print(f"\nIntervention Plan:")
    print(f"  Priority Areas: {len(intervention['priority_areas'])}")
    print(f"  Timeline: {list(intervention['timeline'].keys())}")