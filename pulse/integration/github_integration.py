"""
GitHub Integration Module

Provides comprehensive GitHub API integration including:
- Repository analysis and metrics
- Issue and pull request management
- Workflow automation
- Webhook handling
- Code quality analysis
- Contribution tracking

Author: jetgause
Created: 2025-12-09
"""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from aiohttp import web


logger = logging.getLogger(__name__)


class GitHubEventType(Enum):
    """GitHub webhook event types"""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    ISSUES = "issues"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    CREATE = "create"
    DELETE = "delete"
    FORK = "fork"
    STAR = "star"
    WATCH = "watch"
    RELEASE = "release"
    WORKFLOW_RUN = "workflow_run"
    CHECK_RUN = "check_run"
    CHECK_SUITE = "check_suite"


class IssueState(Enum):
    """GitHub issue states"""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class PullRequestState(Enum):
    """GitHub pull request states"""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


@dataclass
class RepositoryMetrics:
    """Repository metrics and statistics"""
    name: str
    owner: str
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0
    open_pull_requests: int = 0
    total_commits: int = 0
    contributors: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    size_kb: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_push_at: Optional[datetime] = None
    default_branch: str = "main"
    description: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "name": self.name,
            "owner": self.owner,
            "stars": self.stars,
            "forks": self.forks,
            "watchers": self.watchers,
            "open_issues": self.open_issues,
            "open_pull_requests": self.open_pull_requests,
            "total_commits": self.total_commits,
            "contributors": self.contributors,
            "languages": self.languages,
            "size_kb": self.size_kb,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_push_at": self.last_push_at.isoformat() if self.last_push_at else None,
            "default_branch": self.default_branch,
            "description": self.description,
            "topics": self.topics,
            "license": self.license
        }


@dataclass
class Issue:
    """GitHub issue representation"""
    number: int
    title: str
    state: str
    body: Optional[str] = None
    author: Optional[str] = None
    assignees: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    comments: int = 0
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary"""
        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "body": self.body,
            "author": self.author,
            "assignees": self.assignees,
            "labels": self.labels,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "comments": self.comments,
            "url": self.url
        }


@dataclass
class PullRequest:
    """GitHub pull request representation"""
    number: int
    title: str
    state: str
    body: Optional[str] = None
    author: Optional[str] = None
    base_branch: str = "main"
    head_branch: Optional[str] = None
    assignees: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    comments: int = 0
    commits: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    mergeable: Optional[bool] = None
    draft: bool = False
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pull request to dictionary"""
        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "body": self.body,
            "author": self.author,
            "base_branch": self.base_branch,
            "head_branch": self.head_branch,
            "assignees": self.assignees,
            "labels": self.labels,
            "reviewers": self.reviewers,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "comments": self.comments,
            "commits": self.commits,
            "additions": self.additions,
            "deletions": self.deletions,
            "changed_files": self.changed_files,
            "mergeable": self.mergeable,
            "draft": self.draft,
            "url": self.url
        }


class GitHubIntegration:
    """
    Comprehensive GitHub API integration for repository management and analysis
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(
        self,
        token: str,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize GitHub integration
        
        Args:
            token: GitHub personal access token or app token
            session: Optional aiohttp session for connection pooling
        """
        self.token = token
        self._session = session
        self._owned_session = session is None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._owned_session and self._session:
            await self._session.close()
            
    @property
    def headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PulseLibrary-GitHubIntegration"
        }
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Make authenticated API request
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            **kwargs: Additional request parameters
            
        Returns:
            Tuple of (status_code, response_data)
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        kwargs.setdefault("headers", {}).update(self.headers)
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
                
                if response.status == 204:  # No content
                    return response.status, {}
                    
                data = await response.json()
                
                if response.status >= 400:
                    logger.error(f"GitHub API error: {response.status} - {data}")
                    
                return response.status, data
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
            
    async def get_repository(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """
        Get repository information
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository data or None if not found
        """
        status, data = await self._request("GET", f"/repos/{owner}/{repo}")
        return data if status == 200 else None
        
    async def get_repository_metrics(
        self,
        owner: str,
        repo: str
    ) -> Optional[RepositoryMetrics]:
        """
        Get comprehensive repository metrics
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            RepositoryMetrics object or None if repository not found
        """
        repo_data = await self.get_repository(owner, repo)
        if not repo_data:
            return None
            
        # Get additional metrics
        languages = await self.get_repository_languages(owner, repo)
        contributors_count = await self.get_contributors_count(owner, repo)
        
        metrics = RepositoryMetrics(
            name=repo_data["name"],
            owner=repo_data["owner"]["login"],
            stars=repo_data.get("stargazers_count", 0),
            forks=repo_data.get("forks_count", 0),
            watchers=repo_data.get("watchers_count", 0),
            open_issues=repo_data.get("open_issues_count", 0),
            size_kb=repo_data.get("size", 0),
            created_at=datetime.fromisoformat(repo_data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(repo_data["updated_at"].replace("Z", "+00:00")),
            last_push_at=datetime.fromisoformat(repo_data["pushed_at"].replace("Z", "+00:00")),
            default_branch=repo_data.get("default_branch", "main"),
            description=repo_data.get("description"),
            topics=repo_data.get("topics", []),
            license=repo_data.get("license", {}).get("name"),
            languages=languages,
            contributors=contributors_count
        )
        
        return metrics
        
    async def get_repository_languages(
        self,
        owner: str,
        repo: str
    ) -> Dict[str, int]:
        """
        Get repository programming languages
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Dictionary of language names to byte counts
        """
        status, data = await self._request("GET", f"/repos/{owner}/{repo}/languages")
        return data if status == 200 else {}
        
    async def get_contributors_count(self, owner: str, repo: str) -> int:
        """
        Get repository contributors count
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Number of contributors
        """
        status, data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contributors",
            params={"per_page": 1, "anon": "true"}
        )
        
        if status != 200:
            return 0
            
        # Try to get count from Link header
        # For simplicity, we'll just return the length of data for now
        # In production, parse the Link header for accurate count
        return len(data) if isinstance(data, list) else 0
        
    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: IssueState = IssueState.OPEN,
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        since: Optional[datetime] = None,
        per_page: int = 30,
        page: int = 1
    ) -> List[Issue]:
        """
        List repository issues
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state filter
            labels: Label filters
            assignee: Assignee filter
            since: Only issues updated after this time
            per_page: Results per page
            page: Page number
            
        Returns:
            List of Issue objects
        """
        params = {
            "state": state.value,
            "per_page": per_page,
            "page": page
        }
        
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee
        if since:
            params["since"] = since.isoformat()
            
        status, data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/issues",
            params=params
        )
        
        if status != 200 or not isinstance(data, list):
            return []
            
        issues = []
        for item in data:
            # Skip pull requests (they appear in issues endpoint)
            if "pull_request" in item:
                continue
                
            issue = Issue(
                number=item["number"],
                title=item["title"],
                state=item["state"],
                body=item.get("body"),
                author=item["user"]["login"],
                assignees=[a["login"] for a in item.get("assignees", [])],
                labels=[l["name"] for l in item.get("labels", [])],
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                closed_at=datetime.fromisoformat(item["closed_at"].replace("Z", "+00:00")) if item.get("closed_at") else None,
                comments=item.get("comments", 0),
                url=item.get("html_url")
            )
            issues.append(issue)
            
        return issues
        
    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        milestone: Optional[int] = None
    ) -> Optional[Issue]:
        """
        Create a new issue
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            assignees: List of usernames to assign
            labels: List of label names
            milestone: Milestone number
            
        Returns:
            Created Issue object or None if failed
        """
        payload = {"title": title}
        
        if body:
            payload["body"] = body
        if assignees:
            payload["assignees"] = assignees
        if labels:
            payload["labels"] = labels
        if milestone:
            payload["milestone"] = milestone
            
        status, data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues",
            json=payload
        )
        
        if status != 201:
            return None
            
        return Issue(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            body=data.get("body"),
            author=data["user"]["login"],
            assignees=[a["login"] for a in data.get("assignees", [])],
            labels=[l["name"] for l in data.get("labels", [])],
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            comments=data.get("comments", 0),
            url=data.get("html_url")
        )
        
    async def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing issue
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            title: New title
            body: New body
            state: New state ('open' or 'closed')
            assignees: New assignees
            labels: New labels
            
        Returns:
            True if successful, False otherwise
        """
        payload = {}
        
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        if assignees is not None:
            payload["assignees"] = assignees
        if labels is not None:
            payload["labels"] = labels
            
        if not payload:
            return False
            
        status, _ = await self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/{issue_number}",
            json=payload
        )
        
        return status == 200
        
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: PullRequestState = PullRequestState.OPEN,
        base: Optional[str] = None,
        head: Optional[str] = None,
        per_page: int = 30,
        page: int = 1
    ) -> List[PullRequest]:
        """
        List repository pull requests
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Pull request state filter
            base: Base branch filter
            head: Head branch filter
            per_page: Results per page
            page: Page number
            
        Returns:
            List of PullRequest objects
        """
        params = {
            "state": state.value,
            "per_page": per_page,
            "page": page
        }
        
        if base:
            params["base"] = base
        if head:
            params["head"] = head
            
        status, data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params=params
        )
        
        if status != 200 or not isinstance(data, list):
            return []
            
        pull_requests = []
        for item in data:
            pr = PullRequest(
                number=item["number"],
                title=item["title"],
                state=item["state"],
                body=item.get("body"),
                author=item["user"]["login"],
                base_branch=item["base"]["ref"],
                head_branch=item["head"]["ref"],
                assignees=[a["login"] for a in item.get("assignees", [])],
                labels=[l["name"] for l in item.get("labels", [])],
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                merged_at=datetime.fromisoformat(item["merged_at"].replace("Z", "+00:00")) if item.get("merged_at") else None,
                closed_at=datetime.fromisoformat(item["closed_at"].replace("Z", "+00:00")) if item.get("closed_at") else None,
                draft=item.get("draft", False),
                url=item.get("html_url")
            )
            pull_requests.append(pr)
            
        return pull_requests
        
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False
    ) -> Optional[PullRequest]:
        """
        Create a new pull request
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Pull request title
            head: Head branch (changes branch)
            base: Base branch (target branch)
            body: Pull request body
            draft: Create as draft PR
            
        Returns:
            Created PullRequest object or None if failed
        """
        payload = {
            "title": title,
            "head": head,
            "base": base,
            "draft": draft
        }
        
        if body:
            payload["body"] = body
            
        status, data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls",
            json=payload
        )
        
        if status != 201:
            return None
            
        return PullRequest(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            body=data.get("body"),
            author=data["user"]["login"],
            base_branch=data["base"]["ref"],
            head_branch=data["head"]["ref"],
            draft=data.get("draft", False),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            url=data.get("html_url")
        )
        
    async def merge_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
        merge_method: str = "merge"
    ) -> bool:
        """
        Merge a pull request
        
        Args:
            owner: Repository owner
            repo: Repository name
            pull_number: Pull request number
            commit_title: Title for merge commit
            commit_message: Message for merge commit
            merge_method: Merge method ('merge', 'squash', or 'rebase')
            
        Returns:
            True if successful, False otherwise
        """
        payload = {"merge_method": merge_method}
        
        if commit_title:
            payload["commit_title"] = commit_title
        if commit_message:
            payload["commit_message"] = commit_message
            
        status, _ = await self._request(
            "PUT",
            f"/repos/{owner}/{repo}/pulls/{pull_number}/merge",
            json=payload
        )
        
        return status == 200
        
    async def add_label_to_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        labels: List[str]
    ) -> bool:
        """
        Add labels to an issue or pull request
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue or PR number
            labels: List of label names to add
            
        Returns:
            True if successful, False otherwise
        """
        status, _ = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/labels",
            json={"labels": labels}
        )
        
        return status == 200
        
    async def create_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str
    ) -> bool:
        """
        Create a comment on an issue or pull request
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue or PR number
            body: Comment body
            
        Returns:
            True if successful, False otherwise
        """
        status, _ = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            json={"body": body}
        )
        
        return status == 201
        
    async def trigger_workflow(
        self,
        owner: str,
        repo: str,
        workflow_id: str,
        ref: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger a GitHub Actions workflow
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            ref: Git ref (branch, tag, or commit SHA)
            inputs: Workflow input parameters
            
        Returns:
            True if successful, False otherwise
        """
        payload = {"ref": ref}
        
        if inputs:
            payload["inputs"] = inputs
            
        status, _ = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            json=payload
        )
        
        return status == 204


class GitHubWebhookHandler:
    """
    GitHub webhook handler for processing GitHub events
    """
    
    def __init__(self, secret: Optional[str] = None):
        """
        Initialize webhook handler
        
        Args:
            secret: Webhook secret for signature verification
        """
        self.secret = secret
        self.handlers: Dict[GitHubEventType, List[callable]] = {}
        
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature
        
        Args:
            payload: Request payload bytes
            signature: X-Hub-Signature-256 header value
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not self.secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True
            
        if not signature:
            return False
            
        # Remove 'sha256=' prefix
        expected_signature = signature.replace("sha256=", "")
        
        # Calculate HMAC
        mac = hmac.new(
            self.secret.encode(),
            msg=payload,
            digestmod=hashlib.sha256
        )
        calculated_signature = mac.hexdigest()
        
        return hmac.compare_digest(calculated_signature, expected_signature)
        
    def register_handler(
        self,
        event_type: GitHubEventType,
        handler: callable
    ):
        """
        Register an event handler
        
        Args:
            event_type: GitHub event type
            handler: Async callable to handle the event
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def on(self, event_type: GitHubEventType):
        """
        Decorator for registering event handlers
        
        Args:
            event_type: GitHub event type
            
        Example:
            @webhook_handler.on(GitHubEventType.PUSH)
            async def handle_push(payload):
                print(f"Push to {payload['repository']['name']}")
        """
        def decorator(func):
            self.register_handler(event_type, func)
            return func
        return decorator
        
    async def handle_webhook(self, request: web.Request) -> web.Response:
        """
        Handle incoming webhook request
        
        Args:
            request: aiohttp Request object
            
        Returns:
            aiohttp Response object
        """
        # Verify content type
        if request.content_type != "application/json":
            return web.Response(status=400, text="Invalid content type")
            
        # Get event type
        event_type_str = request.headers.get("X-GitHub-Event")
        if not event_type_str:
            return web.Response(status=400, text="Missing event type header")
            
        try:
            event_type = GitHubEventType(event_type_str)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type_str}")
            return web.Response(status=200, text="Event type not handled")
            
        # Read and verify payload
        payload_bytes = await request.read()
        signature = request.headers.get("X-Hub-Signature-256", "")
        
        if not self.verify_signature(payload_bytes, signature):
            logger.error("Invalid webhook signature")
            return web.Response(status=401, text="Invalid signature")
            
        # Parse payload
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON payload")
            
        # Dispatch to handlers
        handlers = self.handlers.get(event_type, [])
        if not handlers:
            logger.info(f"No handlers registered for {event_type.value}")
            return web.Response(status=200, text="No handlers")
            
        # Execute handlers concurrently
        tasks = [handler(payload) for handler in handlers]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error handling webhook: {e}", exc_info=True)
            return web.Response(status=500, text="Internal server error")
            
        return web.Response(status=200, text="OK")
        
    def create_app(self, path: str = "/webhook") -> web.Application:
        """
        Create aiohttp application with webhook endpoint
        
        Args:
            path: Webhook endpoint path
            
        Returns:
            Configured aiohttp Application
        """
        app = web.Application()
        app.router.add_post(path, self.handle_webhook)
        return app


# Example usage
if __name__ == "__main__":
    import os
    
    async def example():
        """Example usage of GitHub integration"""
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("GITHUB_TOKEN environment variable not set")
            return
            
        async with GitHubIntegration(token) as gh:
            # Get repository metrics
            metrics = await gh.get_repository_metrics("jetgause", "Library-")
            if metrics:
                print("Repository Metrics:")
                print(f"  Stars: {metrics.stars}")
                print(f"  Forks: {metrics.forks}")
                print(f"  Open Issues: {metrics.open_issues}")
                print(f"  Contributors: {metrics.contributors}")
                print(f"  Languages: {metrics.languages}")
                
            # List open issues
            issues = await gh.list_issues("jetgause", "Library-", state=IssueState.OPEN)
            print(f"\nOpen Issues: {len(issues)}")
            for issue in issues[:5]:
                print(f"  #{issue.number}: {issue.title}")
                
            # List open pull requests
            prs = await gh.list_pull_requests("jetgause", "Library-", state=PullRequestState.OPEN)
            print(f"\nOpen Pull Requests: {len(prs)}")
            for pr in prs[:5]:
                print(f"  #{pr.number}: {pr.title}")
                
    # Run example
    asyncio.run(example())
