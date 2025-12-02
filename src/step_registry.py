"""
Step Registry
Generic registry for job processing steps that can be dynamically added/removed
"""
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepDefinition:
    """Definition of a processing step"""
    step_id: str
    step_name: str
    handler: Callable
    order: int
    required: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"StepDefinition(id={self.step_id}, name={self.step_name}, order={self.order})"


class StepRegistry:
    """Registry for managing processing steps"""
    
    def __init__(self):
        self._steps: Dict[str, StepDefinition] = {}
        logger.info("Initialized StepRegistry")
    
    def register_step(
        self,
        step_id: str,
        step_name: str,
        handler: Callable,
        order: int,
        required: bool = True,
        retry_on_failure: bool = True,
        max_retries: int = 3,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new step
        
        Args:
            step_id: Unique identifier for the step
            step_name: Human-readable name
            handler: Callable that executes the step
            order: Execution order (lower = earlier)
            required: If True, job fails if this step fails
            retry_on_failure: If True, retry failed steps
            max_retries: Maximum retry attempts
            dependencies: List of step IDs that must complete first
            metadata: Additional step metadata
        """
        step_def = StepDefinition(
            step_id=step_id,
            step_name=step_name,
            handler=handler,
            order=order,
            required=required,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        self._steps[step_id] = step_def
        logger.info(f"Registered step: {step_id} ({step_name}) at order {order}")
    
    def unregister_step(self, step_id: str) -> None:
        """Unregister a step"""
        if step_id in self._steps:
            del self._steps[step_id]
            logger.info(f"Unregistered step: {step_id}")
        else:
            logger.warning(f"Attempted to unregister non-existent step: {step_id}")
    
    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get a step definition"""
        return self._steps.get(step_id)
    
    def get_ordered_steps(self) -> List[StepDefinition]:
        """Get all steps in execution order"""
        return sorted(self._steps.values(), key=lambda s: s.order)
    
    def validate_dependencies(self) -> bool:
        """Validate that all step dependencies exist"""
        for step in self._steps.values():
            for dep in step.dependencies:
                if dep not in self._steps:
                    logger.error(f"Step {step.step_id} depends on non-existent step {dep}")
                    return False
        return True
    
    def get_step_count(self) -> int:
        """Get total number of registered steps"""
        return len(self._steps)
    
    def list_step_ids(self) -> List[str]:
        """Get list of all registered step IDs"""
        return list(self._steps.keys())
    
    def clear(self) -> None:
        """Clear all registered steps"""
        self._steps.clear()
        logger.info("Cleared all steps from registry")


# Global registry instance
step_registry = StepRegistry()


# Decorator for easy step registration
def register_step(
    step_id: str,
    step_name: str,
    order: int,
    required: bool = True,
    retry_on_failure: bool = True,
    max_retries: int = 3,
    dependencies: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator to register a function as a processing step
    
    Usage:
        @register_step(
            step_id="extract_text",
            step_name="Extract Text from EPUB",
            order=1,
            required=True
        )
        def extract_text_step(context: StepExecutionContext) -> dict:
            # Implementation
            return {"extracted_text": text}
    """
    def decorator(func: Callable):
        step_registry.register_step(
            step_id=step_id,
            step_name=step_name,
            handler=func,
            order=order,
            required=required,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            dependencies=dependencies,
            metadata=metadata
        )
        return func
    return decorator
