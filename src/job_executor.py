"""
Job Executor
Executes jobs with step tracking and resume support
"""
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from job_state import JobState, JobStepState, StepStatus, StepExecutionContext
from step_registry import step_registry

logger = logging.getLogger(__name__)


class JobExecutor:
    """Executes jobs with step-by-step tracking and resume capability"""
    
    def __init__(self, jobs_base_path: Path):
        self.jobs_base_path = Path(jobs_base_path)
        self.failed_jobs_path = self.jobs_base_path / "failed"
        self.completed_jobs_path = self.jobs_base_path / "completed"
        self.running_jobs_path = self.jobs_base_path / "running"
        self.pending_jobs_path = self.jobs_base_path / "pending"
        
        # Ensure directories exist
        for path in [self.failed_jobs_path, self.completed_jobs_path, 
                     self.running_jobs_path, self.pending_jobs_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def create_job_state(self, job_id: str) -> JobState:
        """Create a new job state with steps from registry"""
        ordered_steps = step_registry.get_ordered_steps()
        
        steps = [
            JobStepState(
                step_id=step_def.step_id,
                step_name=step_def.step_name,
                status=StepStatus.PENDING
            )
            for step_def in ordered_steps
        ]
        
        job_state = JobState(
            job_id=job_id,
            status="pending",
            steps=steps,
            total_steps=len(steps),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Created job state for {job_id} with {len(steps)} steps")
        return job_state
    
    def load_job_state(self, job_id: str, from_failed: bool = False) -> Optional[JobState]:
        """Load job state from disk"""
        if from_failed:
            job_path = self.failed_jobs_path / job_id
        else:
            job_path = self.running_jobs_path / job_id
        
        if not job_path.exists():
            # Try pending path
            job_path = self.pending_jobs_path / job_id
        
        if not job_path.exists():
            logger.warning(f"Job {job_id} not found")
            return None
        
        state_file = job_path / "job_state.json"
        if state_file.exists():
            try:
                job_state = JobState.load(str(state_file))
                logger.info(f"Loaded job state for {job_id}")
                return job_state
            except Exception as e:
                logger.error(f"Failed to load job state for {job_id}: {e}")
                return None
        else:
            logger.warning(f"Job state file not found for {job_id}")
            return None
    
    def save_job_state(self, job_state: JobState) -> None:
        """Save job state to disk"""
        # Determine current location based on status
        if job_state.status == "running":
            job_path = self.running_jobs_path / job_state.job_id
        elif job_state.status == "failed":
            job_path = self.failed_jobs_path / job_state.job_id
        elif job_state.status == "completed":
            job_path = self.completed_jobs_path / job_state.job_id
        else:
            job_path = self.pending_jobs_path / job_state.job_id
        
        job_path.mkdir(parents=True, exist_ok=True)
        state_file = job_path / "job_state.json"
        job_state.save(str(state_file))
        logger.debug(f"Saved job state for {job_state.job_id}")
    
    def load_job_definition(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job definition from disk"""
        # Try all possible locations
        for base_path in [self.pending_jobs_path, self.running_jobs_path, 
                          self.failed_jobs_path, self.completed_jobs_path]:
            job_path = base_path / job_id
            def_file = job_path / "job_definition.json"
            if def_file.exists():
                with open(def_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        logger.warning(f"Job definition not found for {job_id}")
        return None
    
    def execute_job(
        self,
        job_id: str,
        job_data: Optional[Dict[str, Any]] = None,
        resume: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Execute or resume a job
        
        Args:
            job_id: Job identifier
            job_data: Job configuration data
            resume: If True, resume from last completed step
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if job completed successfully, False otherwise
        """
        try:
            if resume:
                logger.info(f"Resuming job {job_id}")
                job_state = self.load_job_state(job_id, from_failed=True)
                if not job_state:
                    logger.error(f"Cannot resume job {job_id}: state not found")
                    return False
                
                # Load job definition if not provided
                if job_data is None:
                    job_data = self.load_job_definition(job_id)
                    if not job_data:
                        logger.error(f"Cannot resume job {job_id}: definition not found")
                        return False
                
                # Move from failed to running
                self._move_job(job_id, self.failed_jobs_path, self.running_jobs_path)
            else:
                logger.info(f"Starting new job {job_id}")
                job_state = self.create_job_state(job_id)
                
                if job_data is None:
                    job_data = self.load_job_definition(job_id)
                    if not job_data:
                        logger.error(f"Cannot start job {job_id}: definition not found")
                        return False
            
            job_state.status = "running"
            
            # Create work directory
            job_path = self.running_jobs_path / job_id
            work_dir = job_path / "work"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Create execution context
            context = StepExecutionContext(
                job_state=job_state,
                job_data=job_data,
                work_dir=str(work_dir)
            )
            
            # Execute steps
            while True:
                next_step = job_state.get_next_step()
                if not next_step:
                    break
                
                step_def = step_registry.get_step(next_step.step_id)
                if not step_def:
                    logger.error(f"Step definition not found: {next_step.step_id}")
                    job_state.mark_step_failed(next_step.step_id, "Step definition not found")
                    break
                
                # Check dependencies
                if not self._check_dependencies(step_def, job_state):
                    error_msg = f"Dependencies not satisfied for step {step_def.step_id}"
                    logger.error(error_msg)
                    job_state.mark_step_failed(next_step.step_id, error_msg)
                    break
                
                # Execute step with retry logic
                success = self._execute_step_with_retry(
                    step_def,
                    next_step,
                    context,
                    job_state
                )
                
                # Save state after each step
                self.save_job_state(job_state)
                
                # Call progress callback
                if progress_callback:
                    try:
                        progress_callback(job_state)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                if not success and step_def.required:
                    logger.error(f"Required step failed: {step_def.step_id}")
                    break
            
            # Determine final status
            if all(s.status == StepStatus.COMPLETED for s in job_state.steps):
                job_state.status = "completed"
                self._move_job(job_id, self.running_jobs_path, self.completed_jobs_path)
                self.save_job_state(job_state)
                logger.info(f"Job {job_id} completed successfully")
                return True
            else:
                job_state.status = "failed"
                job_state.can_resume = True
                self._move_job(job_id, self.running_jobs_path, self.failed_jobs_path)
                self.save_job_state(job_state)
                logger.error(f"Job {job_id} failed")
                return False
        
        except Exception as e:
            logger.exception(f"Error executing job {job_id}: {e}")
            return False
    
    def _execute_step_with_retry(
        self,
        step_def,
        step_state: JobStepState,
        context: StepExecutionContext,
        job_state: JobState
    ) -> bool:
        """Execute a step with retry logic"""
        max_retries = step_def.max_retries if step_def.retry_on_failure else 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing step {step_def.step_id} (attempt {attempt + 1}/{max_retries})")
                job_state.mark_step_started(step_state.step_id)
                
                # Execute the step handler
                result = step_def.handler(context)
                
                # Store result
                context.set_step_result(step_state.step_id, result)
                
                # Mark as completed
                job_state.mark_step_completed(
                    step_state.step_id,
                    metadata={"result": result, "attempts": attempt + 1}
                )
                
                logger.info(f"Step {step_def.step_id} completed successfully")
                return True
            
            except Exception as e:
                logger.error(f"Step {step_def.step_id} failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying step {step_def.step_id}")
                    continue
                else:
                    job_state.mark_step_failed(step_state.step_id, str(e))
                    return False
        
        return False
    
    def _check_dependencies(self, step_def, job_state: JobState) -> bool:
        """Check if all dependencies are satisfied"""
        for dep_id in step_def.dependencies:
            dep_step = job_state.get_step_by_id(dep_id)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    
    def _move_job(self, job_id: str, from_path: Path, to_path: Path) -> None:
        """Move job between directories"""
        source = from_path / job_id
        dest = to_path / job_id
        
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Remove destination if it exists
                import shutil
                shutil.rmtree(dest)
            source.rename(dest)
            logger.debug(f"Moved job {job_id} from {from_path.name} to {to_path.name}")
