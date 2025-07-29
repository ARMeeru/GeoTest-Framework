# GitHub Issues API integration for Phase 5 intelligent bug tracking
# Provides smart failure management with safeguards to prevent issue spam

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    # Record of a test failure for tracking and analysis
    test_name: str
    failure_type: str
    error_message: str
    timestamp: float
    traceback: str
    failure_hash: str
    retry_count: int = 0
    consecutive_failures: int = 1


@dataclass
class GitHubIssue:
    # GitHub issue representation for bug tracking
    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    created_at: str
    updated_at: str
    url: str
    failure_hash: str


@dataclass
class IssueCreationResult:
    # Result of issue creation attempt
    success: bool
    issue: Optional[GitHubIssue]
    message: str
    rate_limited: bool = False


class GitHubAPIClient:
    # GitHub API client with rate limiting and error handling
    
    def __init__(self, token: str, repo_owner: str, repo_name: str):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.repo_url = f"{self.base_url}/repos/{repo_owner}/{repo_name}"
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GeoTest-Framework-Bug-Tracker"
        })
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.rate_limit_remaining = 5000
        self.rate_limit_reset_time = 0
    
    def _handle_rate_limiting(self):
        # Handle GitHub API rate limiting
        current_time = time.time()
        
        # Check if we're within rate limits
        if current_time < self.rate_limit_reset_time and self.rate_limit_remaining <= 10:
            sleep_time = self.rate_limit_reset_time - current_time + 1
            logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        # Minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.5:  # Maximum 2 requests per second
            time.sleep(0.5 - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        # Make API request with rate limiting and error handling
        self._handle_rate_limiting()
        
        url = f"{self.repo_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Update rate limit tracking
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            reset_timestamp = response.headers.get('X-RateLimit-Reset')
            if reset_timestamp:
                self.rate_limit_reset_time = int(reset_timestamp)
            
            if response.status_code == 429:
                logger.warning("Rate limited by GitHub API")
                raise requests.exceptions.HTTPError("Rate limited", response=response)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            raise
    
    def get_existing_issues(self, labels: List[str] = None, state: str = "open") -> List[GitHubIssue]:
        # Get existing issues with optional filtering
        params = {"state": state, "per_page": 100}
        if labels:
            params["labels"] = ",".join(labels)
        
        try:
            response = self._make_request("GET", "/issues", params=params)
            issues_data = response.json()
            
            issues = []
            for issue_data in issues_data:
                if not issue_data.get("pull_request"):  # Exclude pull requests
                    issue = GitHubIssue(
                        number=issue_data["number"],
                        title=issue_data["title"],
                        body=issue_data["body"] or "",
                        state=issue_data["state"],
                        labels=[label["name"] for label in issue_data["labels"]],
                        created_at=issue_data["created_at"],
                        updated_at=issue_data["updated_at"],
                        url=issue_data["html_url"],
                        failure_hash=self._extract_failure_hash(issue_data["body"] or "")
                    )
                    issues.append(issue)
            
            return issues
            
        except Exception as e:
            logger.error(f"Failed to get existing issues: {e}")
            return []
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> IssueCreationResult:
        # Create a new GitHub issue
        data = {
            "title": title,
            "body": body,
            "labels": labels or []
        }
        
        try:
            response = self._make_request("POST", "/issues", json=data)
            issue_data = response.json()
            
            issue = GitHubIssue(
                number=issue_data["number"],
                title=issue_data["title"],
                body=issue_data["body"] or "",
                state=issue_data["state"],
                labels=[label["name"] for label in issue_data["labels"]],
                created_at=issue_data["created_at"],
                updated_at=issue_data["updated_at"],
                url=issue_data["html_url"],
                failure_hash=self._extract_failure_hash(issue_data["body"] or "")
            )
            
            logger.info(f"Created GitHub issue #{issue.number}: {title}")
            return IssueCreationResult(success=True, issue=issue, message="Issue created successfully")
            
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                return IssueCreationResult(success=False, issue=None, message="Rate limited", rate_limited=True)
            else:
                logger.error(f"Failed to create issue: {e}")
                return IssueCreationResult(success=False, issue=None, message=str(e))
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            return IssueCreationResult(success=False, issue=None, message=str(e))
    
    def close_issue(self, issue_number: int, comment: str = None) -> bool:
        # Close an existing issue
        try:
            # Add comment if provided
            if comment:
                self.add_comment(issue_number, comment)
            
            # Close the issue
            data = {"state": "closed"}
            response = self._make_request("PATCH", f"/issues/{issue_number}", json=data)
            
            logger.info(f"Closed GitHub issue #{issue_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close issue #{issue_number}: {e}")
            return False
    
    def add_comment(self, issue_number: int, comment: str) -> bool:
        # Add a comment to an existing issue
        try:
            data = {"body": comment}
            response = self._make_request("POST", f"/issues/{issue_number}/comments", json=data)
            
            logger.info(f"Added comment to issue #{issue_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add comment to issue #{issue_number}: {e}")
            return False
    
    def _extract_failure_hash(self, body: str) -> str:
        # Extract failure hash from issue body
        for line in body.split('\n'):
            if line.startswith("**Failure Hash:**"):
                return line.replace("**Failure Hash:**", "").strip()
        return ""


class SmartBugTracker:
    # Intelligent bug tracking system with safeguards
    
    def __init__(self, config_file: str = "config/bug_tracking.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Initialize GitHub client if configured
        self.github_client = None
        if self._is_github_configured():
            self.github_client = GitHubAPIClient(
                token=self.config["github"]["token"],
                repo_owner=self.config["github"]["repo_owner"],
                repo_name=self.config["github"]["repo_name"]
            )
        
        # Failure tracking
        self.failure_history_file = Path("failure_history.json")
        self.failure_history = self._load_failure_history()
        
        # Daily issue counter for rate limiting
        self.daily_issues_file = Path("daily_issues.json")
        self.daily_issue_count = self._load_daily_issue_count()
    
    def _load_config(self) -> Dict[str, Any]:
        # Load bug tracking configuration
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "github": {
                    "enabled": False,
                    "token": "",
                    "repo_owner": "",
                    "repo_name": "",
                    "labels": ["bug", "automated-test-failure"]
                },
                "failure_tracking": {
                    "consecutive_failures_threshold": 3,
                    "max_issues_per_day": 5,
                    "similar_failure_grouping": True,
                    "auto_close_on_success": True
                },
                "retry_settings": {
                    "max_retries": 3,
                    "retry_delay": 5,
                    "exponential_backoff": True,
                    "skip_assertion_failures": True
                }
            }
    
    def _save_config(self):
        # Save configuration to file
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_failure_history(self) -> Dict[str, FailureRecord]:
        # Load failure history from file
        if self.failure_history_file.exists():
            try:
                with open(self.failure_history_file, 'r') as f:
                    data = json.load(f)
                    return {
                        key: FailureRecord(**record) 
                        for key, record in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load failure history: {e}")
        return {}
    
    def _save_failure_history(self):
        # Save failure history to file
        try:
            data = {
                key: asdict(record) 
                for key, record in self.failure_history.items()
            }
            with open(self.failure_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure history: {e}")
    
    def _load_daily_issue_count(self) -> Dict[str, int]:
        # Load daily issue count for rate limiting
        if self.daily_issues_file.exists():
            try:
                with open(self.daily_issues_file, 'r') as f:
                    data = json.load(f)
                    # Clean old dates
                    today = datetime.now().strftime("%Y-%m-%d")
                    return {today: data.get(today, 0)}
            except Exception as e:
                logger.warning(f"Failed to load daily issue count: {e}")
        return {}
    
    def _save_daily_issue_count(self):
        # Save daily issue count
        try:
            with open(self.daily_issues_file, 'w') as f:
                json.dump(self.daily_issue_count, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save daily issue count: {e}")
    
    def _is_github_configured(self) -> bool:
        # Check if GitHub integration is properly configured
        github_config = self.config.get("github", {})
        return (
            github_config.get("enabled", False) and
            github_config.get("token") and
            github_config.get("repo_owner") and
            github_config.get("repo_name")
        )
    
    def _generate_failure_hash(self, test_name: str, error_message: str, failure_type: str) -> str:
        # Generate unique hash for failure grouping
        content = f"{test_name}_{failure_type}_{error_message[:200]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _can_create_issue_today(self) -> bool:
        # Check if we can create more issues today
        today = datetime.now().strftime("%Y-%m-%d")
        max_issues = self.config["failure_tracking"]["max_issues_per_day"]
        current_count = self.daily_issue_count.get(today, 0)
        return current_count < max_issues
    
    def record_failure(self, test_name: str, error_message: str, traceback: str, failure_type: str = "test_failure"):
        # Record a test failure for tracking
        failure_hash = self._generate_failure_hash(test_name, error_message, failure_type)
        
        # Update or create failure record
        if failure_hash in self.failure_history:
            record = self.failure_history[failure_hash]
            record.consecutive_failures += 1
            record.timestamp = time.time()
            record.retry_count = 0  # Reset retry count for new failure
        else:
            record = FailureRecord(
                test_name=test_name,
                failure_type=failure_type,
                error_message=error_message,
                timestamp=time.time(),
                traceback=traceback,
                failure_hash=failure_hash
            )
            self.failure_history[failure_hash] = record
        
        logger.info(f"Recorded failure for {test_name}: {record.consecutive_failures} consecutive failures")
        
        # Check if we should create an issue
        threshold = self.config["failure_tracking"]["consecutive_failures_threshold"]
        if record.consecutive_failures >= threshold:
            self._handle_consecutive_failures(record)
        
        self._save_failure_history()
    
    def record_success(self, test_name: str):
        # Record a test success and handle auto-closing issues
        # Find matching failure records for this test
        matching_hashes = [
            failure_hash for failure_hash, record in self.failure_history.items()
            if record.test_name == test_name
        ]
        
        for failure_hash in matching_hashes:
            record = self.failure_history[failure_hash]
            if record.consecutive_failures > 0:
                logger.info(f"Test {test_name} passed, resetting failure count")
                record.consecutive_failures = 0
                
                # Auto-close related issues
                if (self.config["failure_tracking"]["auto_close_on_success"] and 
                    self.github_client):
                    self._auto_close_issues(record)
        
        self._save_failure_history()
    
    def _handle_consecutive_failures(self, record: FailureRecord):
        # Handle consecutive failures by creating or updating issues
        if not self._is_github_configured() or not self.github_client:
            logger.warning("GitHub not configured, cannot create issues")
            return
        
        if not self._can_create_issue_today():
            logger.warning("Daily issue limit reached, not creating new issue")
            return
        
        # Check for existing similar issues
        existing_issues = self.github_client.get_existing_issues(
            labels=self.config["github"]["labels"],
            state="open"
        )
        
        similar_issue = None
        if self.config["failure_tracking"]["similar_failure_grouping"]:
            similar_issue = self._find_similar_issue(record, existing_issues)
        
        if similar_issue:
            # Add comment to existing issue
            self._update_existing_issue(similar_issue, record)
        else:
            # Create new issue
            self._create_new_issue(record)
    
    def _find_similar_issue(self, record: FailureRecord, existing_issues: List[GitHubIssue]) -> Optional[GitHubIssue]:
        # Find similar existing issue for grouping
        for issue in existing_issues:
            if (issue.failure_hash == record.failure_hash or
                record.test_name in issue.title):
                return issue
        return None
    
    def _update_existing_issue(self, issue: GitHubIssue, record: FailureRecord):
        # Update existing issue with new failure information
        comment = f"""
## Additional Failure Occurrence

**Test:** `{record.test_name}`
**Timestamp:** {datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Consecutive Failures:** {record.consecutive_failures}

**Error Message:**
```
{record.error_message}
```

**Failure Hash:** {record.failure_hash}

This failure has now occurred {record.consecutive_failures} times consecutively.
"""
        
        success = self.github_client.add_comment(issue.number, comment)
        if success:
            logger.info(f"Updated existing issue #{issue.number} for {record.test_name}")
    
    def _create_new_issue(self, record: FailureRecord):
        # Create new GitHub issue for failure
        title = f"Test Failure: {record.test_name} ({record.consecutive_failures} consecutive failures)"
        
        body = f"""
## Automated Test Failure Report

**Test Name:** `{record.test_name}`
**Failure Type:** {record.failure_type}
**First Detected:** {datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Consecutive Failures:** {record.consecutive_failures}

### Error Details

**Error Message:**
```
{record.error_message}
```

**Stack Trace:**
```
{record.traceback}
```

### Analysis

This issue was automatically created because the test `{record.test_name}` has failed {record.consecutive_failures} consecutive times, exceeding the threshold of {self.config["failure_tracking"]["consecutive_failures_threshold"]} failures.

**Failure Hash:** {record.failure_hash}

---

*This issue was created automatically by the GeoTest Framework bug tracking system. It will be automatically closed when the test passes again.*
"""
        
        result = self.github_client.create_issue(
            title=title,
            body=body,
            labels=self.config["github"]["labels"]
        )
        
        if result.success:
            # Update daily issue count
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_issue_count[today] = self.daily_issue_count.get(today, 0) + 1
            self._save_daily_issue_count()
            
            logger.info(f"Created new issue #{result.issue.number} for {record.test_name}")
        else:
            logger.error(f"Failed to create issue for {record.test_name}: {result.message}")
    
    def _auto_close_issues(self, record: FailureRecord):
        # Auto-close issues when test passes
        existing_issues = self.github_client.get_existing_issues(
            labels=self.config["github"]["labels"],
            state="open"
        )
        
        for issue in existing_issues:
            if (issue.failure_hash == record.failure_hash or
                record.test_name in issue.title):
                
                comment = f"""
## Test Now Passing ✅

The test `{record.test_name}` is now passing successfully. This issue is being automatically closed.

**Resolution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

The bug tracking system will continue to monitor this test for future failures.
"""
                
                success = self.github_client.close_issue(issue.number, comment)
                if success:
                    logger.info(f"Auto-closed issue #{issue.number} for {record.test_name}")
    
    def get_failure_summary(self, hours: int = 24) -> Dict[str, Any]:
        # Get summary of failures in the specified time window
        cutoff_time = time.time() - (hours * 3600)
        
        recent_failures = [
            record for record in self.failure_history.values()
            if record.timestamp >= cutoff_time
        ]
        
        if not recent_failures:
            return {
                "total_failures": 0,
                "unique_tests": 0,
                "most_common_failures": [],
                "failure_types": {}
            }
        
        # Analyze failure patterns
        failure_types = {}
        test_failures = {}
        
        for record in recent_failures:
            # Count by failure type
            failure_types[record.failure_type] = failure_types.get(record.failure_type, 0) + 1
            
            # Count by test
            test_failures[record.test_name] = test_failures.get(record.test_name, 0) + record.consecutive_failures
        
        # Sort most common failures
        most_common = sorted(test_failures.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_failures": len(recent_failures),
            "unique_tests": len(set(record.test_name for record in recent_failures)),
            "most_common_failures": most_common,
            "failure_types": failure_types,
            "consecutive_failures": sum(record.consecutive_failures for record in recent_failures)
        }
    
    def cleanup_old_failures(self, days: int = 30):
        # Clean up old failure records
        cutoff_time = time.time() - (days * 24 * 3600)
        
        old_hashes = [
            failure_hash for failure_hash, record in self.failure_history.items()
            if record.timestamp < cutoff_time and record.consecutive_failures == 0
        ]
        
        for failure_hash in old_hashes:
            del self.failure_history[failure_hash]
        
        if old_hashes:
            logger.info(f"Cleaned up {len(old_hashes)} old failure records")
            self._save_failure_history()


def get_bug_tracker() -> SmartBugTracker:
    # Get global bug tracker instance
    global _bug_tracker_instance
    if '_bug_tracker_instance' not in globals():
        _bug_tracker_instance = SmartBugTracker()
    return _bug_tracker_instance