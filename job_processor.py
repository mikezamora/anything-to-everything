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
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


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
    epub_path: str
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
    ollama_url: str = "http://localhost:11434"
    segment_words: int = 500
    character_config: Optional[str] = None
    emotion_library: Optional[str] = None
    
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
        args = [self.epub_path]
        
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
        
        if self.ollama_url != "http://localhost:11434":
            args.extend(["--ollama-url", self.ollama_url])
        
        if self.character_config:
            args.extend(["--character-config", self.character_config])
        
        if self.emotion_library:
            args.extend(["--emotion-library", self.emotion_library])
        
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
    
    def process_job(self, job_def: JobDefinition) -> JobResult:
        """
        Process a single job
        
        Args:
            job_def (JobDefinition): Job definition
            
        Returns:
            JobResult: Job execution result
        """
        job_id = job_def.job_id
        start_time = datetime.now().isoformat()
        
        print(f"\n{'='*60}")
        print(f"Processing Job: {job_id}")
        print(f"{'='*60}")
        print(f"EPUB: {job_def.epub_path}")
        print(f"Voice: {job_def.voice_ref_path}")
        print(f"Output: {job_def.output_path}")
        print(f"Priority: {job_def.priority}")
        print(f"{'='*60}\n")
        
        # Move job to running
        self.move_job(job_id, "pending", "running")
        
        # Create job-specific work directory
        job_dir = self.jobs_dir / "running" / job_id
        job_work_dir = job_dir / "work"
        job_work_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare command with job-specific work directory
        args = ["uv", "run", "-m", "main"] + job_def.to_command_args(work_dir=str(job_work_dir))
        
        # Create log file
        job_dir = self.jobs_dir / "running" / job_id
        log_file = job_dir / "output.log"
        
        try:
            # Run the job with real-time output to both log file and terminal
            # Use tee-like functionality to duplicate output to both destinations
            log_file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
            
            try:
                log_file_handle.write(f"Job ID: {job_id}\n")
                log_file_handle.write(f"Command: {' '.join(args)}\n")
                log_file_handle.write(f"Start Time: {start_time}\n")
                log_file_handle.write(f"{'='*60}\n\n")
                log_file_handle.flush()
                
                # Let subprocess inherit stdout/stderr directly for real-time output
                # Just write to log file separately
                process = subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    stdout=None,  # Inherit parent's stdout
                    stderr=None,  # Inherit parent's stderr
                )
                
                # Wait for process to complete
                returncode = process.wait()
                
                # Note: Since we're not capturing output, we can't write it to log in real-time
                # The log will just contain metadata
                
                end_time = datetime.now().isoformat()
                log_file_handle.write(f"\n{'='*60}\n")
                log_file_handle.write(f"End Time: {end_time}\n")
                log_file_handle.write(f"Exit Code: {returncode}\n")
                
                # Create a result object that mimics subprocess.run result
                class Result:
                    def __init__(self, returncode):
                        self.returncode = returncode
                
                result = Result(returncode)
                
            finally:
                log_file_handle.close()
            
            # Read log for result
            with open(log_file, 'r', encoding='utf-8') as log:
                output_log = log.read()
            
            # Determine status
            if result.returncode == 0:
                status = JobStatus.COMPLETED
                self.move_job(job_id, "running", "completed")
                print(f"Job {job_id} completed successfully")
            else:
                status = JobStatus.FAILED
                self.move_job(job_id, "running", "failed")
                print(f"Job {job_id} failed with exit code {result.returncode}")
            
            # Create result
            job_result = JobResult(
                job_id=job_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                exit_code=result.returncode,
                output_log=output_log
            )
            
            # Save result
            result_file = self.jobs_dir / status.value / job_id / "result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(job_result.to_dict(), f, indent=2)
            
            return job_result
            
        except Exception as e:
            end_time = datetime.now().isoformat()
            error_message = str(e)
            
            print(f"Job {job_id} failed with error: {error_message}")
            
            # Move to failed
            self.move_job(job_id, "running", "failed")
            
            # Create error result
            job_result = JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                error_message=error_message
            )
            
            # Save result
            result_file = self.jobs_dir / "failed" / job_id / "result.json"
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
