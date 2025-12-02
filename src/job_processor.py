"""
Job Queue Processor for batch audiobook generation
Processes jobs from the jobs/ directory with individual configurations
"""
import os
import json
import uuid
import time
import shutil
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from job_executor import JobExecutor
from job_state import JobState, StepStatus

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobDefinition:
    """Job configuration definition"""
    job_id: str
    source_text_file: str  # Path to source text file (EPUB or PDF)
    output_path: str
    voice_ref_path: Optional[str] = None  # Optional in character mode
    
    # Optional configuration overrides
    format: str = "m4b"
    detect_characters: bool = False
    ollama_character_detection: bool = False
    character_mode: bool = False
    keep_segments: bool = False
    use_ollama: bool = False
    ollama_model: Optional[str] = None
    ollama_url: str = "http://host.docker.internal:11434"
    segment_words: int = 500
    strip_unknown_tokens: bool = True  # Strip problematic tokens for TTS
    character_config: Optional[str] = None
    emotion_library: Optional[str] = None
    emo_audio_prompt: Optional[str] = None  # Emotion reference audio
    
    # Additional settings
    priority: int = 0  # Higher priority jobs processed first
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobDefinition':
        """Create from dictionary"""
        return cls(**data)
    
    def to_command_args(self, work_dir: Optional[str] = None) -> List[str]:
        """
        Convert job definition to command-line arguments
        
        Args:
            work_dir (str, optional): Override work directory for this job
            
        Returns:
            List of command-line arguments
        """
        args = [self.source_text_file]
        
        # Add voice_ref_path only if provided (optional in character mode)
        if self.voice_ref_path:
            args.append(self.voice_ref_path)
        
        args.extend([
            "-o", self.output_path,
            "--format", self.format
        ])
        
        # Add job-specific work directory if provided
        if work_dir:
            args.extend(["--work-dir", work_dir])
        
        if self.detect_characters:
            args.append("--detect-characters")
        
        if self.ollama_character_detection:
            args.append("--ollama-character-detection")
        
        if self.character_mode:
            args.append("--character-mode")
        
        if self.keep_segments:
            args.append("--keep-segments")
        
        if self.use_ollama:
            args.append("--use-ollama")
        
        if self.ollama_model:
            args.extend(["--ollama-model", self.ollama_model])
        
        if self.ollama_url != "http://host.docker.internal:11434":
            args.extend(["--ollama-url", self.ollama_url])
        
        if self.character_config:
            args.extend(["--character-config", self.character_config])
        
        if self.emotion_library:
            args.extend(["--emotion-library", self.emotion_library])
        
        if self.emo_audio_prompt:
            args.extend(["--emo-audio", self.emo_audio_prompt])
        
        # Add segment words parameter
        args.extend(["--segment-words", str(self.segment_words)])
        
        # Add strip unknown tokens parameter
        if not self.strip_unknown_tokens:
            args.append("--disable-strip-unknown-tokens")

        # args.append("--use-deepspeed")
        
        return args


@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    status: JobStatus
    start_time: str
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    output_log: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobResult':
        """Create from dictionary"""
        data['status'] = JobStatus(data['status'])
        return cls(**data)


class JobQueue:
    """Manages job queue and processing"""
    
    def __init__(self, jobs_dir: str = "./jobs", max_retries: int = 3):
        """
        Initialize job queue
        
        Args:
            jobs_dir (str): Directory containing job folders
            max_retries (int): Maximum number of retry attempts for failed jobs
        """
        self.jobs_dir = Path(jobs_dir)
        self.max_retries = max_retries
        
        # Create jobs directory structure
        self.jobs_dir.mkdir(exist_ok=True)
        (self.jobs_dir / "pending").mkdir(exist_ok=True)
        (self.jobs_dir / "running").mkdir(exist_ok=True)
        (self.jobs_dir / "completed").mkdir(exist_ok=True)
        (self.jobs_dir / "failed").mkdir(exist_ok=True)
    
    def create_job(self, job_def: JobDefinition) -> str:
        """
        Create a new job in the queue
        
        Args:
            job_def (JobDefinition): Job definition
            
        Returns:
            str: Job ID
        """
        # Generate job ID if not provided
        if not job_def.job_id:
            job_def.job_id = str(uuid.uuid4())
        
        # Set creation time
        if not job_def.created_at:
            job_def.created_at = datetime.now().isoformat()
        
        # Create job directory
        job_dir = self.jobs_dir / "pending" / job_def.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create work directory for this job
        job_work_dir = job_dir / "work"
        job_work_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration files to job's work directory if they exist
        config_files_to_copy = [
            ('character_config', 'character_voices.json'),
            ('emotion_library', 'emotion_library.json'),
        ]
        
        for attr_name, default_filename in config_files_to_copy:
            source_path = getattr(job_def, attr_name, None)
            if source_path and os.path.exists(source_path):
                dest_path = job_work_dir / os.path.basename(source_path)
                shutil.copy2(source_path, dest_path)
                print(f"  Copied {os.path.basename(source_path)} to job work directory")
        
        # Copy detected_characters.json if it exists in the default work directory
        default_work_dir = Path("./work")
        detected_chars_file = default_work_dir / "detected_characters.json"
        if detected_chars_file.exists():
            dest_chars_file = job_work_dir / "detected_characters.json"
            shutil.copy2(detected_chars_file, dest_chars_file)
            print(f"  Copied detected_characters.json to job work directory")
        
        # Save job definition
        job_file = job_dir / "job_definition.json"
        with open(job_file, 'w', encoding='utf-8') as f:
            json.dump(job_def.to_dict(), f, indent=2)
        
        print(f"Created job {job_def.job_id}")
        return job_def.job_id
    
    def get_pending_jobs(self) -> List[JobDefinition]:
        """
        Get all pending jobs sorted by priority (highest first)
        
        Returns:
            List of job definitions
        """
        pending_dir = self.jobs_dir / "pending"
        jobs = []
        
        if not pending_dir.exists():
            return jobs
        
        for job_dir in pending_dir.iterdir():
            if job_dir.is_dir():
                job_file = job_dir / "job_definition.json"
                if job_file.exists():
                    with open(job_file, 'r', encoding='utf-8') as f:
                        job_data = json.load(f)
                        jobs.append(JobDefinition.from_dict(job_data))
        
        # Sort by priority (highest first), then by creation time
        jobs.sort(key=lambda x: (-x.priority, x.created_at or ""))
        return jobs
    
    def move_job(self, job_id: str, from_status: str, to_status: str) -> None:
        """
        Move job from one status directory to another
        
        Args:
            job_id (str): Job ID
            from_status (str): Current status directory
            to_status (str): Target status directory
        """
        src_dir = self.jobs_dir / from_status / job_id
        dst_dir = self.jobs_dir / to_status / job_id
        
        if src_dir.exists():
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_dir), str(dst_dir))
    
    def process_job(self, job_def: JobDefinition, resume: bool = False) -> JobResult:
        """
        Process a single job with step-based execution and resume support
        
        Args:
            job_def (JobDefinition): Job definition
            resume (bool): Whether to resume from last completed step
            
        Returns:
            JobResult: Job execution result
        """
        import processing_steps  # Register all processing steps
        
        job_id = job_def.job_id
        start_time = datetime.now().isoformat()
        
        print(f"\n{'='*60}")
        print(f"{'Resuming' if resume else 'Processing'} Job: {job_id}")
        print(f"{'='*60}")
        print(f"Source: {job_def.source_text_file}")
        print(f"Voice: {job_def.voice_ref_path}")
        print(f"Output: {job_def.output_path}")
        print(f"Priority: {job_def.priority}")
        print(f"{'='*60}\n")
        
        # Move job to running (if not already there from failed state)
        if not resume:
            self.move_job(job_id, "pending", "running")
        else:
            # For resume, move from failed to running
            try:
                self.move_job(job_id, "failed", "running")
            except:
                # Already in running, that's fine
                pass
        
        # Use JobExecutor for step-based execution
        executor = JobExecutor(self.jobs_dir)
        
        # Execute job with step tracking and resume support
        success = executor.execute_job(
            job_id=job_id,
            job_data=job_def.to_dict(),
            resume=resume
        )
        
        end_time = datetime.now().isoformat()
        
        # Determine status
        if success:
            status = JobStatus.COMPLETED
            self.move_job(job_id, "running", "completed")
            print(f"\n✅ Job {job_id} completed successfully")
        else:
            status = JobStatus.FAILED
            self.move_job(job_id, "running", "failed")
            print(f"\n❌ Job {job_id} failed")
        
        # Get job state for detailed results
        job_state = executor.load_job_state(job_id)
        error_message = None
        if job_state:
            # Find first failed step
            for step in job_state.steps:
                if step.status == StepStatus.FAILED:
                    error_message = step.error
                    break
        
        # Create result
        job_result = JobResult(
            job_id=job_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            exit_code=0 if success else 1,
            error_message=error_message
        )
        
        # Save result
        result_file = self.jobs_dir / status.value / job_id / "result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(job_result.to_dict(), f, indent=2)
        
        return job_result
    
    def process_queue(self, max_jobs: Optional[int] = None, stop_on_error: bool = False) -> List[JobResult]:
        """
        Process all pending jobs in the queue
        
        Args:
            max_jobs (int, optional): Maximum number of jobs to process
            stop_on_error (bool): Stop processing on first error
            
        Returns:
            List of job results
        """
        results = []
        processed = 0
        
        while True:
            # Get pending jobs
            pending_jobs = self.get_pending_jobs()
            
            if not pending_jobs:
                print("\nNo more pending jobs in queue")
                break
            
            if max_jobs and processed >= max_jobs:
                print(f"\nReached maximum job limit ({max_jobs})")
                break
            
            # Process next job
            job = pending_jobs[0]
            result = self.process_job(job)
            results.append(result)
            processed += 1
            
            # Check if we should stop on error
            if stop_on_error and result.status == JobStatus.FAILED:
                print("\nStopping due to job failure (stop_on_error=True)")
                break
        
        # Print summary
        print(f"\n{'='*60}")
        print("Job Queue Processing Summary")
        print(f"{'='*60}")
        print(f"Total jobs processed: {len(results)}")
        
        completed = sum(1 for r in results if r.status == JobStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == JobStatus.FAILED)
        
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        
        return results
    
    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """
        Get status of a specific job
        
        Args:
            job_id (str): Job ID
            
        Returns:
            JobResult if found, None otherwise
        """
        # Check all status directories
        for status in ["pending", "running", "completed", "failed"]:
            result_file = self.jobs_dir / status / job_id / "result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return JobResult.from_dict(json.load(f))
        
        # Check if pending (no result yet)
        job_file = self.jobs_dir / "pending" / job_id / "job_definition.json"
        if job_file.exists():
            return JobResult(
                job_id=job_id,
                status=JobStatus.PENDING,
                start_time="Not started"
            )
        
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> List[str]:
        """
        List all job IDs, optionally filtered by status
        
        Args:
            status (str, optional): Filter by status (pending, running, completed, failed)
            
        Returns:
            List of job IDs
        """
        job_ids = []
        
        statuses = [status] if status else ["pending", "running", "completed", "failed"]
        
        for stat in statuses:
            status_dir = self.jobs_dir / stat
            if status_dir.exists():
                for job_dir in status_dir.iterdir():
                    if job_dir.is_dir():
                        job_ids.append(job_dir.name)
        
        return job_ids
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if cancelled, False if not found or already running
        """
        pending_dir = self.jobs_dir / "pending" / job_id
        
        if pending_dir.exists():
            cancelled_dir = self.jobs_dir / "cancelled"
            cancelled_dir.mkdir(exist_ok=True)
            
            shutil.move(str(pending_dir), str(cancelled_dir / job_id))
            print(f"Cancelled job {job_id}")
            return True
        
        print(f"Cannot cancel job {job_id} (not found or already running)")
        return False
    
    def get_job_state(self, job_id: str) -> Optional[JobState]:
        """
        Get the state of a job including step progress
        
        Args:
            job_id (str): Job ID
            
        Returns:
            JobState if found, None otherwise
        """
        executor = JobExecutor(self.jobs_dir)
        
        # Try loading from any status folder
        for status in ["pending", "running", "completed", "failed"]:
            job_state = executor.load_job_state(job_id, from_failed=(status == "failed"))
            if job_state:
                return job_state
        
        return None
    
    def get_failed_jobs(self) -> List[str]:
        """
        Get all failed job IDs
        
        Returns:
            List of failed job IDs
        """
        return self.list_jobs(status="failed")
    
    def resume_job(self, job_id: str) -> Optional[JobResult]:
        """
        Resume a failed job from the last completed step
        
        Args:
            job_id (str): Job ID to resume
            
        Returns:
            JobResult if job was found and processed, None otherwise
        """
        # Check if job exists in failed directory
        failed_dir = self.jobs_dir / "failed" / job_id
        if not failed_dir.exists():
            print(f"Failed job {job_id} not found")
            logger.warning(f"Attempted to resume non-existent failed job {job_id}")
            return None
        
        # Load job definition
        job_file = failed_dir / "job_definition.json"
        if not job_file.exists():
            print(f"Job definition not found for {job_id}")
            logger.error(f"Job definition missing for failed job {job_id}")
            return None
        
        with open(job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            job_def = JobDefinition.from_dict(job_data)
        
        print(f"\n{'='*60}")
        print(f"Resuming Failed Job: {job_id}")
        print(f"{'='*60}")
        
        # Get current job state to show progress
        executor = JobExecutor(self.jobs_dir)
        job_state = executor.load_job_state(job_id, from_failed=True)
        
        if job_state:
            completed_steps = len(job_state.get_completed_steps())
            total_steps = job_state.total_steps
            progress = job_state.get_progress_percentage()
            
            print(f"Progress: {completed_steps}/{total_steps} steps completed ({progress:.1f}%)")
            print(f"Last Error: {job_state.last_error}")
            
            # Show completed and pending steps
            print(f"\nCompleted Steps:")
            for step in job_state.steps:
                if step.status == StepStatus.COMPLETED:
                    print(f"  ✓ {step.step_name}")
            
            print(f"\nPending Steps:")
            for step in job_state.steps:
                if step.status in [StepStatus.PENDING, StepStatus.FAILED]:
                    status_icon = "✗" if step.status == StepStatus.FAILED else "○"
                    print(f"  {status_icon} {step.step_name}")
        
        print(f"{'='*60}\n")
        
        # Process with resume flag
        return self.process_job(job_def, resume=True)
    
    def process_single_job(self, job_id: str) -> Optional[JobResult]:
        """
        Process a specific job by ID
        
        Args:
            job_id (str): Job ID to process
            
        Returns:
            JobResult if job was found and processed, None otherwise
        """
        # Load job definition
        job_file = self.jobs_dir / "pending" / job_id / "job_definition.json"
        
        if not job_file.exists():
            print(f"Job {job_id} not found in pending queue")
            return None
        
        with open(job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            job_def = JobDefinition.from_dict(job_data)
        
        # Process the job
        return self.process_job(job_def)


def main():
    """Command-line interface for job processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audiobook generation jobs from queue")
    parser.add_argument("--jobs-dir", default="./jobs", help="Jobs directory (default: ./jobs)")
    parser.add_argument("--max-jobs", type=int, help="Maximum number of jobs to process")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop processing on first error")
    parser.add_argument("--list", choices=["pending", "running", "completed", "failed", "all"], 
                       help="List jobs by status")
    parser.add_argument("--status", help="Get status of specific job ID")
    parser.add_argument("--cancel", help="Cancel a pending job by ID")
    parser.add_argument("--resume", help="Resume a failed job by ID")
    parser.add_argument("--resume-all", action="store_true", help="Resume all failed jobs")
    
    args = parser.parse_args()
    
    queue = JobQueue(jobs_dir=args.jobs_dir)
    
    # Handle list command
    if args.list:
        status = None if args.list == "all" else args.list
        jobs = queue.list_jobs(status=status)
        
        if jobs:
            print(f"\nJobs ({args.list}):")
            for job_id in jobs:
                print(f"  - {job_id}")
            print(f"\nTotal: {len(jobs)}")
        else:
            print(f"\nNo {args.list} jobs found")
        return
    
    # Handle status command
    if args.status:
        result = queue.get_job_status(args.status)
        if result:
            print(f"\nJob Status: {args.status}")
            print(f"Status: {result.status.value}")
            print(f"Start Time: {result.start_time}")
            if result.end_time:
                print(f"End Time: {result.end_time}")
            if result.exit_code is not None:
                print(f"Exit Code: {result.exit_code}")
            if result.error_message:
                print(f"Error: {result.error_message}")
        else:
            print(f"\nJob {args.status} not found")
        return
    
    # Handle cancel command
    if args.cancel:
        queue.cancel_job(args.cancel)
        return
    
    # Handle resume command
    if args.resume:
        result = queue.resume_job(args.resume)
        if result and result.status == JobStatus.COMPLETED:
            print(f"\n✓ Job {args.resume} resumed and completed successfully")
            sys.exit(0)
        else:
            print(f"\n✗ Job {args.resume} failed to complete")
            sys.exit(1)
        return
    
    # Handle resume all command
    if args.resume_all:
        failed_jobs = queue.get_failed_jobs()
        if not failed_jobs:
            print("\nNo failed jobs to resume")
            return
        
        print(f"\nResuming {len(failed_jobs)} failed jobs...")
        results = []
        for job_id in failed_jobs:
            result = queue.resume_job(job_id)
            if result:
                results.append(result)
        
        # Print summary
        completed = sum(1 for r in results if r.status == JobStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == JobStatus.FAILED)
        
        print(f"\n{'='*60}")
        print("Resume Summary")
        print(f"{'='*60}")
        print(f"Total resumed: {len(results)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        
        sys.exit(0 if failed == 0 else 1)
        return
    
    # Process queue
    print("Starting job queue processor...")
    print(f"Jobs directory: {queue.jobs_dir.absolute()}")
    print()
    
    results = queue.process_queue(
        max_jobs=args.max_jobs,
        stop_on_error=args.stop_on_error
    )
    
    # Return exit code based on results
    if any(r.status == JobStatus.FAILED for r in results):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
