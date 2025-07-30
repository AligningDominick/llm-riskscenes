"""Unit tests for CLI module."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, Mock
import json
import pandas as pd

from lmse.cli import cli, evaluate, analyze, list_scenarios, init_config, list_models


class TestCLI:
    """Test cases for CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_cli_version(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower()
    
    def test_cli_help(self, runner):
        """Test help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'LLM Multilingual Safety Evaluation' in result.output
        assert 'evaluate' in result.output
        assert 'analyze' in result.output
    
    @patch('lmse.cli.ModelLoader')
    @patch('lmse.cli.SafetyEvaluator')
    def test_evaluate_command(self, mock_evaluator, mock_loader, runner):
        """Test evaluate command."""
        # Mock model and evaluator
        mock_model = Mock()
        mock_loader.load.return_value = mock_model
        
        mock_results = pd.DataFrame({
            'scenario_id': ['s1', 's2'],
            'safety_score': [80, 90],
            'risk_level': ['LOW', 'LOW'],
            'compliance_score': [0.8, 0.9]
        })
        mock_evaluator.return_value.evaluate.return_value = mock_results
        mock_evaluator.return_value.generate_report.return_value = 'report.html'
        
        # Run command
        result = runner.invoke(evaluate, [
            '--model', 'mock',
            '--languages', 'english',
            '--languages', 'spanish',
            '--domains', 'healthcare'
        ])
        
        assert result.exit_code == 0
        assert 'Loading model: mock' in result.output
        assert 'Starting evaluation' in result.output
        
        # Verify calls
        mock_loader.load.assert_called_once_with('mock', mock_evaluator.return_value.config)
        mock_evaluator.return_value.evaluate.assert_called_once()
    
    def test_evaluate_missing_model(self, runner):
        """Test evaluate command without required model."""
        result = runner.invoke(evaluate, [])
        assert result.exit_code != 0
        assert 'Missing option' in result.output
    
    @patch('lmse.cli.pd.read_csv')
    @patch('lmse.cli.SafetyEvaluator')
    def test_analyze_command(self, mock_evaluator, mock_read_csv, runner, temp_dir):
        """Test analyze command."""
        # Mock data
        mock_df = pd.DataFrame({
            'scenario_id': ['s1'],
            'safety_score': [80],
            'risk_level': ['LOW']
        })
        mock_read_csv.return_value = mock_df
        
        mock_analysis = {
            'summary': {
                'total_evaluations': 1,
                'overall_safety_score': 80,
                'pass_rate': 1.0
            },
            'recommendations': ['Test recommendation']
        }
        mock_evaluator.return_value.analyze.return_value = mock_analysis
        
        # Run command
        result = runner.invoke(analyze, [
            '--results', 'test.csv',
            '--output', str(temp_dir / 'analysis.json'),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert 'Analyzing Results' in result.output
        assert 'Total Evaluations: 1' in result.output
    
    @patch('lmse.cli.ScenarioLoader')
    def test_list_scenarios_command(self, mock_loader, runner):
        """Test list-scenarios command."""
        # Mock scenarios
        mock_scenarios = [
            {
                'id': 's1',
                'language': 'english',
                'domain': 'healthcare',
                'tags': ['test']
            },
            {
                'id': 's2',
                'language': 'spanish',
                'domain': 'legal',
                'tags': ['test']
            }
        ]
        mock_loader.return_value.load_scenarios.return_value = mock_scenarios
        
        # Test table format
        result = runner.invoke(list_scenarios, ['--format', 'table'])
        assert result.exit_code == 0
        assert 'Available Scenarios' in result.output
        assert 's1' in result.output
        assert 'english' in result.output
        
        # Test JSON format
        result = runner.invoke(list_scenarios, ['--format', 'json'])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert len(output) == 2
        assert output[0]['id'] == 's1'
    
    def test_init_config_command(self, runner, temp_dir):
        """Test init-config command."""
        config_path = temp_dir / 'test_config.yaml'
        
        result = runner.invoke(init_config, [
            '--output', str(config_path),
            '--format', 'yaml'
        ])
        
        assert result.exit_code == 0
        assert 'Configuration created' in result.output
        assert config_path.exists()
        
        # Check content
        with open(config_path) as f:
            content = f.read()
            assert 'evaluation:' in content
            assert 'api_keys:' in content
    
    @patch('lmse.cli.ModelLoader')
    def test_list_models_command(self, mock_loader, runner):
        """Test list-models command."""
        mock_loader.list_available_models.return_value = {
            'claude-3-opus': 'Claude 3 Opus model',
            'gpt-4': 'OpenAI GPT-4',
            'mock': 'Mock model for testing'
        }
        
        result = runner.invoke(list_models, [])
        assert result.exit_code == 0
        assert 'Available Models' in result.output
        assert 'claude-3-opus' in result.output
        assert 'gpt-4' in result.output
        assert 'mock' in result.output
    
    def test_evaluate_with_config(self, runner, temp_dir):
        """Test evaluate command with config file."""
        # Create config file
        config_path = temp_dir / 'config.yaml'
        config_data = {
            'evaluation': {'batch_size': 5},
            'output': {'directory': str(temp_dir)}
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch('lmse.cli.ModelLoader') as mock_loader:
            with patch('lmse.cli.SafetyEvaluator') as mock_evaluator:
                mock_loader.load.return_value = Mock()
                mock_evaluator.return_value.evaluate.return_value = pd.DataFrame()
                mock_evaluator.return_value.generate_report.return_value = 'report.html'
                
                result = runner.invoke(evaluate, [
                    '--config', str(config_path),
                    '--model', 'mock'
                ])
                
                assert result.exit_code == 0
    
    def test_evaluate_sync_mode(self, runner):
        """Test evaluate command in sync mode."""
        with patch('lmse.cli.ModelLoader') as mock_loader:
            with patch('lmse.cli.SafetyEvaluator') as mock_evaluator:
                mock_loader.load.return_value = Mock()
                mock_evaluator.return_value.evaluate.return_value = pd.DataFrame()
                mock_evaluator.return_value.generate_report.return_value = 'report.html'
                
                result = runner.invoke(evaluate, [
                    '--model', 'mock',
                    '--sync'
                ])
                
                assert result.exit_code == 0
                # Verify sync mode was set
                config_calls = [call for call in mock_evaluator.call_args_list]
                # Config should have async=False