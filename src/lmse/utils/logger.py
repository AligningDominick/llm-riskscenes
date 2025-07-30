"""Logging utilities for the evaluation framework."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logger(
    name: str,
    level: Union[str, int] = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class EvaluationLogger:
    """Specialized logger for evaluation sessions."""
    
    def __init__(self, session_id: str, output_dir: Path = Path("logs")):
        """Initialize evaluation logger.
        
        Args:
            session_id: Unique session identifier
            output_dir: Directory for log files
        """
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"evaluation_{timestamp}_{session_id}.log"
        
        # Setup loggers
        self.main_logger = setup_logger(
            f"lmse.evaluation.{session_id}",
            level="INFO",
            log_file=log_file
        )
        
        self.error_logger = setup_logger(
            f"lmse.errors.{session_id}",
            level="ERROR",
            log_file=self.output_dir / f"errors_{timestamp}_{session_id}.log"
        )
        
        self.metrics_logger = setup_logger(
            f"lmse.metrics.{session_id}",
            level="INFO",
            log_file=self.output_dir / f"metrics_{timestamp}_{session_id}.log",
            format_string="%(asctime)s - %(message)s"
        )
    
    def log_start(self, config: dict):
        """Log evaluation session start.
        
        Args:
            config: Session configuration
        """
        self.main_logger.info(f"Starting evaluation session: {self.session_id}")
        self.main_logger.info(f"Configuration: {config}")
    
    def log_scenario(self, scenario_id: str, language: str, domain: str):
        """Log scenario evaluation start.
        
        Args:
            scenario_id: Scenario identifier
            language: Scenario language
            domain: Scenario domain
        """
        self.main_logger.info(
            f"Evaluating scenario: {scenario_id} "
            f"(language={language}, domain={domain})"
        )
    
    def log_response(self, scenario_id: str, model: str, response_length: int):
        """Log model response.
        
        Args:
            scenario_id: Scenario identifier
            model: Model name
            response_length: Length of response
        """
        self.main_logger.debug(
            f"Received response for {scenario_id} from {model} "
            f"(length={response_length})"
        )
    
    def log_error(self, scenario_id: str, error: Exception):
        """Log evaluation error.
        
        Args:
            scenario_id: Scenario identifier
            error: Exception that occurred
        """
        self.error_logger.error(
            f"Error evaluating {scenario_id}: {type(error).__name__}: {error}",
            exc_info=True
        )
    
    def log_metrics(self, metrics: dict):
        """Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.metrics_logger.info(f"{key}: {value}")
    
    def log_completion(self, total_scenarios: int, duration: float):
        """Log evaluation completion.
        
        Args:
            total_scenarios: Total scenarios evaluated
            duration: Evaluation duration in seconds
        """
        self.main_logger.info(
            f"Evaluation completed: {total_scenarios} scenarios "
            f"in {duration:.2f} seconds"
        )
        self.main_logger.info(f"Session {self.session_id} finished")


class ProgressLogger:
    """Logger for tracking evaluation progress."""
    
    def __init__(self, total: int, logger: Optional[logging.Logger] = None):
        """Initialize progress logger.
        
        Args:
            total: Total number of items
            logger: Optional logger to use
        """
        self.total = total
        self.current = 0
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress.
        
        Args:
            increment: Number of items completed
            message: Optional progress message
        """
        self.current += increment
        progress = (self.current / self.total) * 100
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        log_msg = f"Progress: {self.current}/{self.total} ({progress:.1f}%)"
        log_msg += f" - Rate: {rate:.1f}/s - ETA: {eta:.0f}s"
        
        if message:
            log_msg += f" - {message}"
        
        self.logger.info(log_msg)
    
    def complete(self):
        """Mark progress as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Completed {self.total} items in {elapsed:.1f} seconds "
            f"({self.total/elapsed:.1f} items/s)"
        )