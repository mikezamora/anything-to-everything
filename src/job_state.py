"""
Job State Management
Tracks the state of jobs and their individual steps for resumable execution
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json


class StepStatus(str, Enum):
    """Step execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class JobStepState:
    """State of an individual job step"""
    step_id: str
    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobStepState':
        """Create from dictionary"""
        data['status'] = StepStatus(data['status'])
        return cls(**data)


@dataclass
class JobState:
    """Complete state of a job"""
    job_id: str
    status: str
    steps: List[JobStepState]
    current_step_index: int = 0
    total_steps: int = 0
    can_resume: bool = False
    last_error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        """Initialize after creation"""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()
        if not self.total_steps:
            self.total_steps = len(self.steps)
    
    def get_next_step(self) -> Optional[JobStepState]:
        """Get the next step to execute"""
        for step in self.steps:
            if step.status in [StepStatus.PENDING, StepStatus.FAILED]:
                return step
        return None
    
    def mark_step_started(self, step_id: str) -> None:
        """Mark a step as started"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = StepStatus.IN_PROGRESS
                step.started_at = datetime.utcnow().isoformat()
                break
        self.updated_at = datetime.utcnow().isoformat()
    
    def mark_step_completed(self, step_id: str, metadata: Optional[Dict] = None) -> None:
        """Mark a step as completed"""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.utcnow().isoformat()
                if metadata:
                    step.metadata.update(metadata)
                self.current_step_index = i + 1
                break
        self.updated_at = datetime.utcnow().isoformat()
    
    def mark_step_failed(self, step_id: str, error: str) -> None:
        """Mark a step as failed"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = StepStatus.FAILED
                step.error = error
                step.retry_count += 1
                break
        self.last_error = error
        self.can_resume = True
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_completed_steps(self) -> List[JobStepState]:
        """Get all completed steps"""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def get_progress_percentage(self) -> float:
        """Calculate job progress percentage"""
        completed = len([s for s in self.steps if s.status == StepStatus.COMPLETED])
        return (completed / self.total_steps) * 100 if self.total_steps > 0 else 0
    
    def get_step_by_id(self, step_id: str) -> Optional[JobStepState]:
        """Get a step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'steps': [step.to_dict() for step in self.steps],
            'current_step_index': self.current_step_index,
            'total_steps': self.total_steps,
            'can_resume': self.can_resume,
            'last_error': self.last_error,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobState':
        """Create from dictionary"""
        data['steps'] = [JobStepState.from_dict(s) for s in data.get('steps', [])]
        return cls(**data)
    
    def save(self, filepath: str) -> None:
        """Save state to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'JobState':
        """Load state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class StepExecutionContext:
    """Context passed to step handlers during execution"""
    job_state: JobState
    job_data: Dict[str, Any]
    work_dir: str
    step_results: Dict[str, Any] = field(default_factory=dict)
    
    def get_previous_step_result(self, step_id: str) -> Optional[Any]:
        """Get result from a previous step"""
        return self.step_results.get(step_id)
    
    def set_step_result(self, step_id: str, result: Any) -> None:
        """Store result from current step"""
        self.step_results[step_id] = result
    
    def get_step_metadata(self, step_id: str) -> Dict[str, Any]:
        """Get metadata from a completed step"""
        step = self.job_state.get_step_by_id(step_id)
        return step.metadata if step else {}
