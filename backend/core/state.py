"""
Application State

Creates a singleton PipelineManager instance that is
shared across all API routes during application runtime.
"""

from backend.core.pipeline_manager import PipelineManager

# Global pipeline manager shared throughout the backend.
manager = PipelineManager()
