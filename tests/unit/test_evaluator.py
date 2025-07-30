"""Unit tests for SafetyEvaluator."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from lmse.core.evaluator import SafetyEvaluator
from lmse.models.base import BaseModel


class TestSafetyEvaluator:
    """Test cases for SafetyEvaluator class."""
    
    def test_init_with_config(self, test_config):
        """Test initialization with config object."""
        evaluator = SafetyEvaluator(test_config)
        assert evaluator.config == test_config
        assert evaluator.results == []
        assert evaluator.session_id is not None
    
    def test_init_with_dict(self):
        """Test initialization with dictionary."""
        config_dict = {"evaluation": {"async": False}}
        evaluator = SafetyEvaluator(config_dict)
        assert evaluator.config.get("evaluation.async") is False
    
    def test_evaluate_sync(self, test_config, mock_model, sample_scenarios):
        """Test synchronous evaluation."""
        test_config.set("evaluation.async", False)
        evaluator = SafetyEvaluator(test_config)
        
        # Mock scenario loader
        with patch.object(evaluator.scenario_loader, 'load_scenarios', return_value=sample_scenarios):
            results = evaluator.evaluate(mock_model)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_scenarios)
        assert all(col in results.columns for col in [
            'scenario_id', 'language', 'domain', 'response',
            'risk_score', 'risk_level', 'safety_score'
        ])
    
    @pytest.mark.asyncio
    async def test_evaluate_async(self, test_config, mock_model, sample_scenarios):
        """Test asynchronous evaluation."""
        test_config.set("evaluation.async", True)
        evaluator = SafetyEvaluator(test_config)
        
        # Mock scenario loader
        with patch.object(evaluator.scenario_loader, 'load_scenarios', return_value=sample_scenarios):
            results = evaluator.evaluate(mock_model, async_batch_size=2)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_scenarios)
    
    def test_evaluate_with_filters(self, test_config, mock_model):
        """Test evaluation with language and domain filters."""
        evaluator = SafetyEvaluator(test_config)
        
        # Mock scenario loader to verify filters are passed correctly
        with patch.object(evaluator.scenario_loader, 'load_scenarios') as mock_loader:
            mock_loader.return_value = []
            
            evaluator.evaluate(
                mock_model,
                languages=["english", "spanish"],
                domains=["healthcare", "legal"]
            )
            
            mock_loader.assert_called_once_with(
                ["english", "spanish"],
                ["healthcare", "legal"]
            )
    
    def test_evaluate_with_custom_scenarios(self, test_config, mock_model, sample_scenarios):
        """Test evaluation with custom scenarios."""
        evaluator = SafetyEvaluator(test_config)
        
        results = evaluator.evaluate(
            mock_model,
            scenarios=sample_scenarios
        )
        
        assert len(results) == len(sample_scenarios)
    
    def test_evaluate_error_handling(self, test_config, sample_scenarios):
        """Test error handling during evaluation."""
        evaluator = SafetyEvaluator(test_config)
        
        # Create a model that raises errors
        error_model = Mock(spec=BaseModel)
        error_model.name = "error-model"
        error_model.generate.side_effect = Exception("Test error")
        
        with patch.object(evaluator.scenario_loader, 'load_scenarios', return_value=sample_scenarios):
            results = evaluator.evaluate(error_model)
        
        # Should still return results with error markers
        assert len(results) == len(sample_scenarios)
        assert all(results['risk_level'] == 'ERROR')
        assert all(results['risk_score'] == -1)
    
    def test_analyze_basic(self, test_config):
        """Test basic analysis functionality."""
        evaluator = SafetyEvaluator(test_config)
        
        # Create sample results
        results_data = {
            'scenario_id': ['s1', 's2', 's3'],
            'language': ['english', 'spanish', 'english'],
            'domain': ['healthcare', 'legal', 'healthcare'],
            'safety_score': [80, 60, 90],
            'compliance_score': [0.8, 0.6, 0.9],
            'risk_level': ['LOW', 'MEDIUM', 'LOW'],
            'risk_score': [20, 40, 10]
        }
        results_df = pd.DataFrame(results_data)
        
        analysis = evaluator.analyze(results_df)
        
        assert 'summary' in analysis
        assert 'by_language' in analysis
        assert 'by_domain' in analysis
        assert 'recommendations' in analysis
        
        # Check summary
        summary = analysis['summary']
        assert summary['total_evaluations'] == 3
        assert summary['overall_safety_score'] == pytest.approx(76.67, 0.01)
    
    def test_analyze_by_language(self, test_config):
        """Test language-specific analysis."""
        evaluator = SafetyEvaluator(test_config)
        
        results_data = {
            'scenario_id': ['s1', 's2', 's3', 's4'],
            'language': ['english', 'english', 'spanish', 'spanish'],
            'domain': ['healthcare', 'legal', 'healthcare', 'legal'],
            'safety_score': [80, 85, 60, 65],
            'compliance_score': [0.8, 0.85, 0.6, 0.65],
            'risk_level': ['LOW', 'LOW', 'HIGH', 'MEDIUM'],
            'risk_score': [20, 15, 60, 35]
        }
        results_df = pd.DataFrame(results_data)
        
        analysis = evaluator.analyze(results_df)
        lang_analysis = analysis['by_language']
        
        assert 'english' in lang_analysis
        assert 'spanish' in lang_analysis
        assert lang_analysis['english']['avg_safety_score'] == 82.5
        assert lang_analysis['spanish']['avg_safety_score'] == 62.5
    
    def test_generate_recommendations(self, test_config):
        """Test recommendation generation."""
        evaluator = SafetyEvaluator(test_config)
        
        # Create results with various issues
        results_data = {
            'scenario_id': ['s1', 's2', 's3'],
            'language': ['hindi', 'hindi', 'hindi'],
            'domain': ['healthcare', 'healthcare', 'healthcare'],
            'safety_score': [40, 45, 50],  # Low scores
            'compliance_score': [0.4, 0.45, 0.5],
            'risk_level': ['CRITICAL', 'CRITICAL', 'HIGH'],
            'risk_score': [80, 75, 60],
            'identified_issues': [
                ['Provides specific medical advice'],
                ['Provides specific medical advice'],
                ['Lacks disclaimer']
            ]
        }
        results_df = pd.DataFrame(results_data)
        
        analysis = evaluator.analyze(results_df)
        recommendations = analysis['recommendations']
        
        assert len(recommendations) > 0
        assert any('safety score is below' in rec for rec in recommendations)
        assert any('hindi' in rec.lower() for rec in recommendations)
    
    def test_save_results(self, test_config, temp_dir):
        """Test saving evaluation results."""
        test_config.set("output.directory", str(temp_dir))
        evaluator = SafetyEvaluator(test_config)
        
        # Create sample results
        results_data = {
            'scenario_id': ['s1', 's2'],
            'language': ['english', 'spanish'],
            'domain': ['healthcare', 'legal'],
            'safety_score': [80, 70],
            'risk_level': ['LOW', 'MEDIUM']
        }
        results_df = pd.DataFrame(results_data)
        
        # Save results
        evaluator._save_results(results_df)
        
        # Check files were created
        csv_files = list(temp_dir.glob("*.csv"))
        json_files = list(temp_dir.glob("*.json"))
        
        assert len(csv_files) >= 1
        assert len(json_files) >= 1
    
    def test_generate_report(self, test_config, temp_dir):
        """Test report generation."""
        test_config.set("output.directory", str(temp_dir / "reports"))
        evaluator = SafetyEvaluator(test_config)
        
        # Set some sample results
        evaluator.results = [
            {
                'scenario_id': 's1',
                'language': 'english',
                'domain': 'healthcare',
                'safety_score': 80,
                'compliance_score': 0.8,
                'cultural_score': 0.9,
                'risk_level': 'LOW',
                'risk_score': 20
            }
        ]
        
        with patch('lmse.utils.report_generator.ReportGenerator') as mock_generator:
            report_path = evaluator.generate_report()
            
            assert report_path is not None
            mock_generator.return_value.generate.assert_called_once()
    
    def test_session_id_format(self, test_config):
        """Test session ID format."""
        evaluator = SafetyEvaluator(test_config)
        
        # Session ID should be timestamp format
        assert len(evaluator.session_id) == 15  # YYYYMMDD_HHMMSS
        assert evaluator.session_id[8] == '_'
        
        # Should be parseable as datetime
        datetime.strptime(evaluator.session_id, "%Y%m%d_%H%M%S")