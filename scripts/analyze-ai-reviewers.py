#!/usr/bin/env python3
"""
LLM-as-Judge Analysis of AI Code Reviewers

This script performs semantic analysis of AI code review comments to identify
truly unique insights vs. rephrased duplicates across reviewers.

Approach:
1. Fetch PR comments from GitHub using gh CLI
2. Extract discrete concerns using LLM
3. Semantically cluster similar concerns across reviewers
4. Identify truly unique insights per reviewer
5. Generate analysis report

Prerequisites:
- gh CLI installed and authenticated
- Authentication (one of):
  - GOOGLE_API_KEY environment variable (AI Studio key)
  - GOOGLE_APPLICATION_CREDENTIALS for Vertex AI
  - gcloud application-default credentials
- Python 3.11+

Usage:
    python scripts/analyze-ai-reviewers.py --repo owner/repo [--prs N] [--output FILE]

    # With API key:
    GOOGLE_API_KEY=your-key python scripts/analyze-ai-reviewers.py --repo owner/repo

    # With Vertex AI (uses ADC):
    python scripts/analyze-ai-reviewers.py --vertex --project your-project --repo owner/repo

Author: https://github.com/qredek
License: MIT
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


# Rate limiting configuration
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests
RATE_LIMIT_BUFFER = 100  # Pause when fewer than this many requests remain
RATE_LIMIT_PAUSE = 60  # Seconds to pause when approaching rate limit
_last_request_time: float = 0.0

# Global repo variable (set from args)
REPO: str = ""

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google import genai as vertexai_genai
    from google.genai import types as genai_types
except ImportError:
    vertexai_genai = None

if not genai and not vertexai_genai:
    print("Error: No Google AI SDK found.")
    print("Install with: pip install google-generativeai")
    print("Or for Vertex AI: pip install google-genai")
    sys.exit(1)


# Model configuration
DEFAULT_MODEL = "gemini-2.0-flash"

# AI Reviewer bot identifiers - add your own reviewers here
AI_REVIEWERS = {
    "claude": ["claude[bot]", "@claude"],
    "gemini": ["gemini-code-assist[bot]", "google-gemini-code-assist[bot]"],
    "cursor": ["cursor[bot]"],
    "codex": ["chatgpt-codex-connector[bot]", "openai[bot]"],
    "coderabbit": ["coderabbitai[bot]"],
    "sourcery": ["sourcery-ai[bot]"],
    "deepsource": ["deepsource-autofix[bot]"],
}


class Severity(str, Enum):
    """Severity levels for code review concerns."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    """Categories for code review concerns."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TYPE_SAFETY = "type_safety"
    ERROR_HANDLING = "error_handling"
    RACE_CONDITION = "race_condition"
    NULL_SAFETY = "null_safety"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CODE_STYLE = "code_style"
    ARCHITECTURE = "architecture"
    OTHER = "other"


@dataclass
class Concern:
    """A discrete concern extracted from a code review comment."""
    summary: str
    severity: Severity
    category: Category
    reviewer: str
    pr_number: int
    original_text: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ConcernCluster:
    """A cluster of semantically similar concerns."""
    representative_summary: str
    concerns: list[Concern] = field(default_factory=list)
    reviewers: set[str] = field(default_factory=set)

    @property
    def is_unique(self) -> bool:
        """True if only one reviewer raised this concern."""
        return len(self.reviewers) == 1

    @property
    def unique_reviewer(self) -> Optional[str]:
        """Return the single reviewer if unique, else None."""
        return next(iter(self.reviewers)) if self.is_unique else None


@dataclass
class ReviewerStats:
    """Statistics for a single AI reviewer."""
    name: str
    total_comments: int = 0
    total_concerns: int = 0
    unique_concerns: int = 0
    concerns_by_severity: dict[str, int] = field(default_factory=dict)
    concerns_by_category: dict[str, int] = field(default_factory=dict)
    unique_catches: list[str] = field(default_factory=list)


# =============================================================================
# Serialization and Anonymization
# =============================================================================

def concern_to_dict(concern: Concern) -> dict:
    """Convert a Concern dataclass to a dictionary."""
    return {
        "summary": concern.summary,
        "severity": concern.severity.value,
        "category": concern.category.value,
        "reviewer": concern.reviewer,
        "pr_number": concern.pr_number,
        "original_text": concern.original_text,
        "file_path": concern.file_path,
        "line_number": concern.line_number,
    }


def cluster_to_dict(cluster: ConcernCluster) -> dict:
    """Convert a ConcernCluster dataclass to a dictionary."""
    return {
        "representative_summary": cluster.representative_summary,
        "concerns": [concern_to_dict(c) for c in cluster.concerns],
        "reviewers": sorted(cluster.reviewers),
        "is_unique": cluster.is_unique,
    }


def stats_to_dict(stats: ReviewerStats) -> dict:
    """Convert a ReviewerStats dataclass to a dictionary."""
    return {
        "name": stats.name,
        "total_comments": stats.total_comments,
        "total_concerns": stats.total_concerns,
        "unique_concerns": stats.unique_concerns,
        "concerns_by_severity": stats.concerns_by_severity,
        "concerns_by_category": stats.concerns_by_category,
        "unique_catches": stats.unique_catches,
    }


class Anonymizer:
    """Consistent anonymization of analysis data using deterministic mapping."""

    def __init__(self, salt: str = ""):
        self.salt = salt
        self._pr_map: dict[int, str] = {}
        self._file_map: dict[str, str] = {}
        self._pr_counter = 0
        self._file_counter = 0

    def anonymize_pr(self, pr_number: int) -> str:
        """Map PR number to sequential ID."""
        if pr_number not in self._pr_map:
            self._pr_counter += 1
            self._pr_map[pr_number] = f"PR-{self._pr_counter:03d}"
        return self._pr_map[pr_number]

    def anonymize_file(self, file_path: str | None) -> str | None:
        """Map file path to hashed ID, preserving extension."""
        if not file_path:
            return None
        if file_path not in self._file_map:
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            self._file_counter += 1
            base = f"file-{self._file_counter:03d}"
            self._file_map[file_path] = f"{base}.{ext}" if ext else base
        return self._file_map[file_path]

    def anonymize_concern(self, concern_dict: dict) -> dict:
        """Anonymize a single concern dictionary."""
        return {
            "summary": concern_dict["summary"],
            "severity": concern_dict["severity"],
            "category": concern_dict["category"],
            "reviewer": concern_dict["reviewer"],
            "pr_number": self.anonymize_pr(concern_dict["pr_number"]),
            "original_text": None,
            "file_path": self.anonymize_file(concern_dict.get("file_path")),
            "line_number": None,
        }


def generalize_summaries_with_llm(
    model,
    model_type: str,
    concerns: list[dict]
) -> list[dict]:
    """Use LLM to strip code-specific details from concern summaries."""
    if not concerns:
        return concerns

    summaries = [c["summary"] for c in concerns]

    prompt = f"""Generalize these code review concern summaries to remove:
- Specific function/variable/class names
- File paths or module names
- Line numbers or code snippets
- Project-specific terminology

Keep the core issue type and severity clear.

Examples:
- "Missing null check in getUserEmail()" -> "Missing null check in function return value"
- "Race condition in AuthController.validate()" -> "Race condition in validation logic"
- "XSS vulnerability in src/components/UserInput.tsx" -> "XSS vulnerability in user input component"

Summaries to generalize:
{json.dumps(summaries, indent=2)}

Return a JSON array of generalized summaries in the same order."""

    try:
        response_text = generate_content(model, model_type, prompt)
        generalized = json.loads(response_text)

        # Validate response is a list of strings with correct length
        if not isinstance(generalized, list):
            print("  Warning: LLM returned non-list response, skipping generalization")
            return concerns
        if len(generalized) != len(concerns):
            print(f"  Warning: LLM returned {len(generalized)} summaries for {len(concerns)} concerns")
            return concerns
        if not all(isinstance(s, str) for s in generalized):
            print("  Warning: LLM returned non-string summaries, skipping generalization")
            return concerns

        result = []
        for concern, new_summary in zip(concerns, generalized):
            updated = concern.copy()
            updated["summary"] = new_summary
            result.append(updated)
        return result

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Failed to generalize summaries: {type(e).__name__}")
        return concerns


def export_analysis_data(
    concerns: list[Concern],
    clusters: list[ConcernCluster],
    stats: dict[str, ReviewerStats],
    report: str,
    repo: str,
    prs_analyzed: int,
    anonymize: bool = False,
    anonymize_summaries: bool = False,
    model=None,
    model_type: str = None,
) -> dict:
    """Export analysis data to a dictionary for JSON serialization.

    Args:
        concerns: All extracted concerns.
        clusters: Clustered concerns.
        stats: Per-reviewer statistics.
        report: Generated markdown report.
        repo: Repository name.
        prs_analyzed: Number of PRs analyzed.
        anonymize: Whether to anonymize identifiers.
        anonymize_summaries: Whether to LLM-generalize summaries.
        model: LLM model (required if anonymize_summaries=True).
        model_type: Model type (required if anonymize_summaries=True).

    Returns:
        Dictionary ready for JSON serialization.
    """
    concern_dicts = [concern_to_dict(c) for c in concerns]
    cluster_dicts = [cluster_to_dict(c) for c in clusters]
    stats_dicts = {name: stats_to_dict(s) for name, s in stats.items()}

    if anonymize:
        anonymizer = Anonymizer()

        concern_dicts = [anonymizer.anonymize_concern(c) for c in concern_dicts]

        anon_clusters = []
        for cluster in cluster_dicts:
            anon_cluster = {
                "representative_summary": cluster["representative_summary"],
                "concerns": [anonymizer.anonymize_concern(c) for c in cluster["concerns"]],
                "reviewers": cluster["reviewers"],
                "is_unique": cluster["is_unique"],
            }
            anon_clusters.append(anon_cluster)
        cluster_dicts = anon_clusters

        for stat in stats_dicts.values():
            if anonymize_summaries:
                stat["unique_catches"] = []

        repo = "anonymized-repository"

    if anonymize_summaries and model:
        print("  Generalizing concern summaries with LLM...")
        # Collect all unique summaries to generalize once for consistency
        all_summaries = set(c["summary"] for c in concern_dicts)
        for cluster in cluster_dicts:
            all_summaries.add(cluster["representative_summary"])
            for c in cluster["concerns"]:
                all_summaries.add(c["summary"])

        # Generalize all unique summaries in one call
        unique_summaries = list(all_summaries)
        dummy_concerns = [{"summary": s} for s in unique_summaries]
        generalized = generalize_summaries_with_llm(model, model_type, dummy_concerns)

        # Build mapping from original to generalized
        summary_map = {orig: gen["summary"] for orig, gen in zip(unique_summaries, generalized)}

        # Apply mapping consistently to all concerns
        for c in concern_dicts:
            c["summary"] = summary_map.get(c["summary"], c["summary"])

        for cluster in cluster_dicts:
            cluster["representative_summary"] = summary_map.get(
                cluster["representative_summary"], cluster["representative_summary"]
            )
            for c in cluster["concerns"]:
                c["summary"] = summary_map.get(c["summary"], c["summary"])

    return {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "metadata": {
            "repository": repo,
            "prs_analyzed": prs_analyzed,
            "total_concerns": len(concerns),
            "total_clusters": len(clusters),
            "anonymized": anonymize,
        },
        "concerns": concern_dicts,
        "clusters": cluster_dicts,
        "stats": stats_dicts,
        "report": report if not anonymize else None,
    }


def create_model(use_vertex: bool = False, project: str | None = None):
    """Create a generative model using API key or Vertex AI with ADC.

    Args:
        use_vertex: If True, use Vertex AI with Application Default Credentials.
        project: GCP project ID (required for Vertex AI if GCP_PROJECT_ID not set).

    Returns:
        A tuple of (model/client, model_type).
    """
    if use_vertex:
        if not vertexai_genai:
            print("Error: google-genai package not found for Vertex AI.")
            print("Install with: pip install google-genai")
            sys.exit(1)

        # Check for project - require explicit setting, no hidden defaults
        if not project:
            project = os.environ.get("GCP_PROJECT_ID")
        if not project:
            print("Error: GCP project ID required for Vertex AI.")
            print("Set GCP_PROJECT_ID environment variable or use --project flag.")
            sys.exit(1)

        print(f"Using Vertex AI with project: {project}, model: {DEFAULT_MODEL}")
        client = vertexai_genai.Client(
            vertexai=True,
            project=project,
            location="us-central1"
        )
        return client, "vertex"
    else:
        if not genai:
            print("Error: google-generativeai package not found.")
            print("Install with: pip install google-generativeai")
            sys.exit(1)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            print("Get an API key from: https://aistudio.google.com/")
            print("Or use --vertex flag to use Vertex AI with ADC.")
            sys.exit(1)

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(DEFAULT_MODEL), "genai"


def identify_reviewer(author: str) -> Optional[str]:
    """Identify which AI reviewer authored a comment."""
    author_lower = author.lower()
    for reviewer, identifiers in AI_REVIEWERS.items():
        for identifier in identifiers:
            if identifier.lower() in author_lower:
                return reviewer
    return None


def check_rate_limit() -> tuple[int, int]:
    """Check GitHub API rate limit status.

    Returns:
        Tuple of (remaining requests, reset timestamp).
    """
    try:
        result = subprocess.run(
            ["gh", "api", "rate_limit", "--jq", ".rate"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        return data.get("remaining", 5000), data.get("reset", 0)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"\n  Warning: Could not check GitHub rate limit due to {type(e).__name__}. Assuming a low limit to be safe.")
        return 10, 0  # Assume a very low limit to trigger a pause if needed


def rate_limited_api_call(cmd: list[str], check_limit: bool = True) -> subprocess.CompletedProcess:
    """Execute a GitHub API call with rate limiting.

    Adds a minimum delay between requests and checks rate limit status
    periodically to avoid hitting limits.

    Args:
        cmd: The command to execute.
        check_limit: Whether to check rate limit before calling.

    Returns:
        The completed process result.
    """
    global _last_request_time

    # Enforce minimum interval between requests
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    # Periodically check rate limit
    if check_limit and _last_request_time > 0:
        remaining, reset_time = check_rate_limit()
        if remaining < RATE_LIMIT_BUFFER:
            wait_time = max(reset_time - time.time(), RATE_LIMIT_PAUSE)
            print(f"\n  Rate limit low ({remaining} remaining). Pausing for {int(wait_time)}s...")
            time.sleep(wait_time)

    _last_request_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for rate limit error and retry after waiting
    if result.returncode != 0 and "rate limit" in (result.stderr or "").lower():
        print("\n  Rate limited. Waiting 60s before retry...")
        time.sleep(RATE_LIMIT_PAUSE)
        _last_request_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)

    return result


def fetch_prs_with_ai_reviews(limit: int = 50) -> list[int]:
    """Fetch recent PR numbers that have AI review comments.

    For large limits (>100), fetches in batches to avoid GitHub API timeouts.
    """
    print(f"Fetching recent PRs with AI reviews (limit={limit})...")

    pr_numbers = []
    batch_size = 100  # GitHub's practical limit per request

    # Fetch in batches to avoid timeout
    batch = 0
    seen_prs = set()

    while len(pr_numbers) < limit:
        batch += 1
        print(f"  Fetching batch {batch}...")

        # Use gh search for better pagination support
        # Search for PRs, sorted by update date (merged PRs include closed ones)
        cmd = [
            "gh", "search", "prs",
            "--repo", REPO,
            "--merged",  # Get merged PRs (most have AI reviews)
            "--limit", str(batch_size),
            "--sort", "updated",
            "--order", "desc",
            "--json", "number"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            batch_prs = json.loads(result.stdout)

            if not batch_prs:
                print("  No more PRs found")
                break

            new_count = 0
            for pr in batch_prs:
                pr_num = pr.get("number")
                if pr_num and pr_num not in seen_prs:
                    seen_prs.add(pr_num)
                    pr_numbers.append(pr_num)
                    new_count += 1
                    if len(pr_numbers) >= limit:
                        break

            print(f"  Found {new_count} new PRs (total: {len(pr_numbers)})")

            # gh search doesn't have true cursor pagination, so we stop after first batch
            # For larger datasets, we need to use gh api with pagination
            if len(batch_prs) < batch_size or batch >= 1:
                break

        except subprocess.CalledProcessError as e:
            print(f"Error fetching PRs (batch {batch}): {e.stderr}")
            if batch == 1:
                return []
            break

    # If we need more PRs, fall back to direct API call with pagination
    if len(pr_numbers) < limit:
        print("  Fetching additional PRs via API...")
        pr_numbers = fetch_prs_via_api(limit, seen_prs)

    print(f"Found {len(pr_numbers)} PRs total")
    return pr_numbers[:limit]


def fetch_prs_via_api(limit: int, existing: set[int]) -> list[int]:
    """Fetch PRs using direct GitHub API calls with pagination."""
    pr_numbers = list(existing)
    page = 1
    per_page = 100

    while len(pr_numbers) < limit:
        cmd = [
            "gh", "api",
            f"repos/{REPO}/pulls",
            "-X", "GET",
            "-f", "state=all",
            "-f", f"per_page={per_page}",
            "-f", f"page={page}",
            "-f", "sort=updated",
            "-f", "direction=desc",
            "--jq", ".[].number"
        ]

        # Check rate limit every 5 pages to avoid excessive checking
        result = rate_limited_api_call(cmd, check_limit=(page % 5 == 1))
        if result.returncode != 0:
            print(f"Error fetching PRs (page {page}): {result.stderr}")
            break

        if not result.stdout.strip():
            break

        batch_numbers = [int(n) for n in result.stdout.strip().split("\n") if n]

        if not batch_numbers:
            break

        for pr_num in batch_numbers:
            if pr_num not in existing:
                existing.add(pr_num)
                pr_numbers.append(pr_num)
                if len(pr_numbers) >= limit:
                    break

        print(f"  Page {page}: found {len(batch_numbers)} PRs (total: {len(pr_numbers)})")
        page += 1

        if len(batch_numbers) < per_page:
            break

    return pr_numbers


def fetch_pr_comments(pr_number: int, check_rate_limit: bool = False) -> list[dict]:
    """Fetch all comments from a PR including review comments.

    Args:
        pr_number: The PR number to fetch comments for.
        check_rate_limit: Whether to check rate limit before API calls.

    Returns:
        List of comment dictionaries.
    """
    comments = []

    # Fetch PR comments (general discussion)
    cmd_comments = [
        "gh", "api",
        f"repos/{REPO}/issues/{pr_number}/comments",
        "--jq", "."
    ]

    # Fetch review comments (inline on code)
    cmd_reviews = [
        "gh", "api",
        f"repos/{REPO}/pulls/{pr_number}/comments",
        "--jq", "."
    ]

    try:
        # General comments - use rate-limited call
        result = rate_limited_api_call(cmd_comments, check_limit=check_rate_limit)
        if result.returncode != 0:
            print(f"  Warning: Failed to fetch PR comments for #{pr_number}: {result.stderr[:100] if result.stderr else 'unknown error'}")
        elif result.stdout.strip():
            pr_comments = json.loads(result.stdout)
            for c in pr_comments:
                c["comment_type"] = "pr_comment"
            comments.extend(pr_comments)
    except json.JSONDecodeError as e:
        print(f"  Warning: Invalid JSON in PR comments for #{pr_number}: {e}")

    try:
        # Review comments (inline) - use rate-limited call
        result = rate_limited_api_call(cmd_reviews, check_limit=False)  # Already checked above
        if result.returncode != 0:
            print(f"  Warning: Failed to fetch review comments for #{pr_number}: {result.stderr[:100] if result.stderr else 'unknown error'}")
        elif result.stdout.strip():
            review_comments = json.loads(result.stdout)
            for c in review_comments:
                c["comment_type"] = "review_comment"
            comments.extend(review_comments)
    except json.JSONDecodeError as e:
        print(f"  Warning: Invalid JSON in review comments for #{pr_number}: {e}")

    return comments


def filter_ai_comments(comments: list[dict]) -> dict[str, list[dict]]:
    """Filter comments by AI reviewer."""
    ai_comments: dict[str, list[dict]] = {
        reviewer: [] for reviewer in AI_REVIEWERS.keys()
    }

    for comment in comments:
        # Handle null user field (deleted/ghost users) - .get default only applies to missing keys
        author = (comment.get("user") or {}).get("login", "")
        reviewer = identify_reviewer(author)
        if reviewer:
            ai_comments[reviewer].append(comment)

    return ai_comments


def generate_content(model, model_type: str, prompt: str) -> str:
    """Generate content using the appropriate API based on model type.

    Args:
        model: The model or client instance.
        model_type: Either 'vertex' or 'genai'.
        prompt: The prompt to send.

    Returns:
        The generated text response.
    """
    if model_type == "vertex":
        # Vertex AI (google-genai SDK)
        response = model.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        return response.text
    else:
        # Google AI (google-generativeai SDK)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        return response.text


def extract_concerns_with_llm(
    model,
    model_type: str,
    comments: list[dict],
    reviewer: str,
    pr_number: int
) -> list[Concern]:
    """Use LLM to extract discrete concerns from reviewer comments."""
    if not comments:
        return []

    # Prepare comment text for analysis
    comment_texts = []
    for i, c in enumerate(comments):
        body = c.get("body", "")
        file_path = c.get("path", "")
        line = c.get("line", "")

        text = f"[Comment {i+1}]"
        if file_path:
            text += f" File: {file_path}"
        if line:
            text += f" Line: {line}"
        text += f"\n{body}"
        comment_texts.append(text)

    combined_text = "\n\n---\n\n".join(comment_texts)

    prompt = f"""Analyze these code review comments from the "{reviewer}" AI reviewer on PR #{pr_number}.

Extract each DISCRETE concern or suggestion. For each concern:
1. Summarize in one clear sentence what the concern is
2. Assign severity: critical (will cause bugs/security issues), high (likely problems), medium (should fix), low (nice to have)
3. Assign category: bug, security, performance, type_safety, error_handling, race_condition, null_safety, testing, documentation, code_style, architecture, other

IMPORTANT:
- Each concern should be a distinct issue, not a rephrasing of another
- Skip meta-comments like "I reviewed this PR" or "Overall looks good"
- Focus on actionable technical feedback

Comments to analyze:
{combined_text}

Respond with a JSON array of concerns. Example format:
[
  {{
    "summary": "Missing null check before accessing user.email",
    "severity": "high",
    "category": "null_safety",
    "original_snippet": "The first few words of the original comment..."
  }}
]

If there are no actionable concerns, return an empty array: []
"""

    try:
        response_text = generate_content(model, model_type, prompt)

        # Parse response
        concerns_data = json.loads(response_text)
        concerns = []

        for cd in concerns_data:
            try:
                concern = Concern(
                    summary=cd.get("summary", ""),
                    severity=Severity(cd.get("severity", "medium").lower()),
                    category=Category(cd.get("category", "other").lower()),
                    reviewer=reviewer,
                    pr_number=pr_number,
                    original_text=cd.get("original_snippet", ""),
                )
                concerns.append(concern)
            except (ValueError, KeyError):
                continue

        return concerns

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse concerns JSON for {reviewer} on PR #{pr_number}: {e}")
        return []
    except Exception as e:
        print(f"  Warning: Failed to extract concerns for {reviewer} on PR #{pr_number}: {type(e).__name__}: {e}")
        return []


def cluster_concerns_for_pr(
    model,
    model_type: str,
    pr_concerns: list[Concern],
    pr_number: int
) -> list[ConcernCluster]:
    """Cluster concerns within a single PR to find overlapping reviews.

    This determines if multiple reviewers caught the same issue on the same PR.
    """
    if not pr_concerns:
        return []

    # If only one reviewer, each concern is unique
    reviewers_present = set(c.reviewer for c in pr_concerns)
    if len(reviewers_present) <= 1:
        return [
            ConcernCluster(
                representative_summary=c.summary,
                concerns=[c],
                reviewers={c.reviewer}
            )
            for c in pr_concerns
        ]

    # Multiple reviewers - use LLM to find overlapping concerns
    concern_summaries = []
    for i, c in enumerate(pr_concerns):
        concern_summaries.append(f"{i}: [{c.reviewer}] {c.summary}")

    concerns_text = "\n".join(concern_summaries)

    prompt = f"""Analyze these code review concerns from different AI reviewers on PR #{pr_number}.

Concerns (format: index: [reviewer] summary):
{concerns_text}

Group concerns that are about THE SAME underlying issue, even if phrased differently.
- Two concerns about "missing error handling in function X" should be grouped
- A concern about "add try/catch" and "handle exceptions" for the same code should be grouped
- Different concerns about different issues should NOT be grouped

Respond with JSON:
{{
  "clusters": [
    {{
      "representative_summary": "Clear one-line description of the shared concern",
      "concern_indices": [0, 3, 5]
    }}
  ]
}}

IMPORTANT: Every concern index (0 to {len(pr_concerns)-1}) must appear in exactly one cluster.
Single-concern clusters are fine for truly unique insights.
"""

    try:
        response_text = generate_content(model, model_type, prompt)
        cluster_data = json.loads(response_text)
        clusters = []
        assigned_indices: set[int] = set()

        for cd in cluster_data.get("clusters", []):
            cluster = ConcernCluster(
                representative_summary=cd.get("representative_summary", "Unknown"),
            )
            for idx in cd.get("concern_indices", []):
                if idx in assigned_indices:
                    continue
                if 0 <= idx < len(pr_concerns):
                    assigned_indices.add(idx)
                    concern = pr_concerns[idx]
                    cluster.concerns.append(concern)
                    cluster.reviewers.add(concern.reviewer)
            if cluster.concerns:
                clusters.append(cluster)

        # Add fallback clusters for any missing concerns
        missing_indices = set(range(len(pr_concerns))) - assigned_indices
        if missing_indices:
            print(f"  Warning: LLM did not assign {len(missing_indices)} concerns for PR #{pr_number}. Adding as individual clusters.")
        for idx in sorted(missing_indices):
            c = pr_concerns[idx]
            clusters.append(ConcernCluster(
                representative_summary=c.summary,
                concerns=[c],
                reviewers={c.reviewer}
            ))

        return clusters

    except (json.JSONDecodeError, Exception):
        # Fallback: each concern is its own cluster
        return [
            ConcernCluster(
                representative_summary=c.summary,
                concerns=[c],
                reviewers={c.reviewer}
            )
            for c in pr_concerns
        ]


def cluster_concerns_with_llm(
    model,
    model_type: str,
    all_concerns: list[Concern]
) -> list[ConcernCluster]:
    """Cluster concerns per-PR to find overlapping reviews.

    Groups concerns by PR first, then clusters within each PR.
    This is more meaningful than global clustering since we want to know
    if multiple reviewers caught the same issue on the same PR.
    """
    if not all_concerns:
        return []

    # Group concerns by PR
    concerns_by_pr: dict[int, list[Concern]] = {}
    for concern in all_concerns:
        if concern.pr_number not in concerns_by_pr:
            concerns_by_pr[concern.pr_number] = []
        concerns_by_pr[concern.pr_number].append(concern)

    # Cluster within each PR
    all_clusters: list[ConcernCluster] = []
    pr_count = len(concerns_by_pr)
    multi_reviewer_prs = 0

    for i, (pr_number, pr_concerns) in enumerate(concerns_by_pr.items()):
        reviewers_present = set(c.reviewer for c in pr_concerns)
        if len(reviewers_present) > 1:
            multi_reviewer_prs += 1
            # Only call LLM for PRs with multiple reviewers
            pr_clusters = cluster_concerns_for_pr(model, model_type, pr_concerns, pr_number)
        else:
            # Single reviewer - each concern is unique
            pr_clusters = [
                ConcernCluster(
                    representative_summary=c.summary,
                    concerns=[c],
                    reviewers={c.reviewer}
                )
                for c in pr_concerns
            ]
        all_clusters.extend(pr_clusters)

        # Progress indicator for multi-reviewer PRs
        if (i + 1) % 50 == 0:
            print(f"  Clustered {i + 1}/{pr_count} PRs...")

    print(f"  Processed {pr_count} PRs ({multi_reviewer_prs} with multiple reviewers)")
    return all_clusters


def compute_reviewer_stats(
    clusters: list[ConcernCluster],
    comment_counts: dict[str, int]
) -> dict[str, ReviewerStats]:
    """Compute statistics for each reviewer."""
    stats: dict[str, ReviewerStats] = {}

    for reviewer in AI_REVIEWERS.keys():
        stats[reviewer] = ReviewerStats(
            name=reviewer,
            total_comments=comment_counts.get(reviewer, 0),
            concerns_by_severity={s.value: 0 for s in Severity},
            concerns_by_category={c.value: 0 for c in Category}
        )

    # Count concerns per reviewer
    for cluster in clusters:
        for concern in cluster.concerns:
            reviewer_stat = stats[concern.reviewer]
            reviewer_stat.total_concerns += 1
            reviewer_stat.concerns_by_severity[concern.severity.value] += 1
            reviewer_stat.concerns_by_category[concern.category.value] += 1

        # Track unique concerns
        if cluster.is_unique:
            reviewer = cluster.unique_reviewer
            if reviewer:
                stats[reviewer].unique_concerns += 1
                stats[reviewer].unique_catches.append(cluster.representative_summary)

    return stats


def generate_report(
    stats: dict[str, ReviewerStats],
    clusters: list[ConcernCluster],
    prs_analyzed: int,
    repo: str,
    anonymize: bool = False
) -> str:
    """Generate markdown report of analysis."""
    today = date.today().isoformat()
    display_repo = "anonymized-repository" if anonymize else repo

    # Get active reviewers (those with comments)
    active_reviewers = [r for r in stats.keys() if stats[r].total_comments > 0]

    lines = [
        "# LLM-as-Judge Analysis Results",
        "",
        f"**Repository**: {display_repo}",
        f"**Date**: {today}",
        f"**PRs Analyzed**: {prs_analyzed}",
        f"**Total Concerns Extracted**: {sum(s.total_concerns for s in stats.values())}",
        f"**Unique Concern Clusters**: {len(clusters)}",
        "",
        "## Summary by Reviewer",
        "",
        "| Reviewer | Comments | Concerns | Unique Catches | Unique Rate |",
        "|----------|----------|----------|----------------|-------------|",
    ]

    for reviewer in active_reviewers:
        s = stats[reviewer]
        unique_rate = f"{(s.unique_concerns / s.total_concerns * 100):.1f}%" if s.total_concerns > 0 else "N/A"
        lines.append(f"| **{reviewer.title()}** | {s.total_comments} | {s.total_concerns} | {s.unique_concerns} | {unique_rate} |")

    lines.extend([
        "",
        "## Concern Severity Distribution",
        "",
        "| Reviewer | Critical | High | Medium | Low |",
        "|----------|----------|------|--------|-----|",
    ])

    for reviewer in active_reviewers:
        s = stats[reviewer]
        sev = s.concerns_by_severity
        lines.append(f"| **{reviewer.title()}** | {sev.get('critical', 0)} | {sev.get('high', 0)} | {sev.get('medium', 0)} | {sev.get('low', 0)} |")

    lines.extend([
        "",
        "## Concern Category Distribution",
        "",
        "| Reviewer | Bug | Security | Performance | Type Safety | Error Handling |",
        "|----------|-----|----------|-------------|-------------|----------------|",
    ])

    for reviewer in active_reviewers:
        s = stats[reviewer]
        cat = s.concerns_by_category
        lines.append(
            f"| **{reviewer.title()}** | {cat.get('bug', 0)} | {cat.get('security', 0)} | "
            f"{cat.get('performance', 0)} | {cat.get('type_safety', 0)} | {cat.get('error_handling', 0)} |"
        )

    # Unique catches by reviewer
    lines.extend([
        "",
        "## Unique Catches by Reviewer",
        "",
        "Issues found by ONLY that reviewer (not caught by others on the same PR):",
        "",
    ])

    for reviewer in active_reviewers:
        s = stats[reviewer]
        lines.append(f"### {reviewer.title()} ({s.unique_concerns} unique)")
        lines.append("")
        if s.unique_catches:
            for catch in s.unique_catches[:10]:  # Top 10
                lines.append(f"- {catch}")
        else:
            lines.append("- No unique catches found")
        lines.append("")

    # Overlapping concerns
    overlap_clusters = [c for c in clusters if not c.is_unique]
    lines.extend([
        "## Overlapping Concerns",
        "",
        f"Concerns raised by multiple reviewers: **{len(overlap_clusters)}**",
        "",
    ])

    if overlap_clusters:
        for cluster in overlap_clusters[:10]:  # Top 10
            reviewers = ", ".join(sorted(cluster.reviewers))
            lines.append(f"- **{cluster.representative_summary}** ({reviewers})")

    # Recommendations
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])

    # Find highest unique catch rate
    active_stats = [stats[r] for r in active_reviewers if stats[r].total_concerns > 0]
    if active_stats:
        best_unique = max(active_stats, key=lambda s: s.unique_concerns)
        best_signal = max(
            active_stats,
            key=lambda s: s.unique_concerns / s.total_concerns if s.total_concerns > 0 else 0
        )

        lines.extend([
            f"1. **Highest unique catch count**: {best_unique.name.title()} ({best_unique.unique_concerns} unique concerns)",
            f"2. **Highest signal-to-noise ratio**: {best_signal.name.title()} "
            f"({best_signal.unique_concerns}/{best_signal.total_concerns} concerns are unique)",
            "",
            "### Reviewer Value Assessment",
            "",
        ])

        for reviewer in active_reviewers:
            s = stats[reviewer]
            if s.total_concerns == 0:
                continue
            unique_pct = (s.unique_concerns / s.total_concerns * 100)
            critical_high = s.concerns_by_severity.get("critical", 0) + s.concerns_by_severity.get("high", 0)

            if unique_pct > 30 and critical_high > 0:
                assessment = "HIGH VALUE - catches unique critical/high issues"
            elif unique_pct > 20:
                assessment = "GOOD VALUE - decent unique catch rate"
            elif s.total_concerns > 50 and unique_pct < 10:
                assessment = "REVIEW NEEDED - high volume, low unique value"
            else:
                assessment = "MODERATE VALUE"

            lines.append(f"- **{reviewer.title()}**: {assessment}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    global REPO

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge analysis of AI code reviewers"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        required=True,
        help="GitHub repository (owner/name)"
    )
    parser.add_argument(
        "--prs", "-n",
        type=int,
        default=20,
        help="Number of PRs to analyze (default: 20)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for report (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        help="Use Vertex AI with Application Default Credentials"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="GCP project ID for Vertex AI"
    )
    parser.add_argument(
        "--save-data",
        type=str,
        default=None,
        metavar="FILE",
        help="Save extracted analysis data to JSON file"
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize data when saving (requires --save-data)"
    )
    parser.add_argument(
        "--anonymize-summaries",
        action="store_true",
        help="Use LLM to generalize concern summaries (slower, more thorough)"
    )
    args = parser.parse_args()

    # Set global repo
    REPO = args.repo

    # Initialize model (API key or Vertex AI with ADC)
    model, model_type = create_model(use_vertex=args.vertex, project=args.project)

    print("=" * 60)
    print("LLM-as-Judge Analysis of AI Code Reviewers")
    print(f"Repository: {REPO}")
    print("=" * 60)
    print()

    # Fetch PRs
    pr_numbers = fetch_prs_with_ai_reviews(limit=args.prs)
    if not pr_numbers:
        print("No PRs found to analyze.")
        return

    # Collect all concerns
    all_concerns: list[Concern] = []
    comment_counts: dict[str, int] = {r: 0 for r in AI_REVIEWERS.keys()}

    for i, pr_number in enumerate(pr_numbers):
        print(f"[{i+1}/{len(pr_numbers)}] Analyzing PR #{pr_number}...")

        # Fetch comments - check rate limit every 20 PRs
        check_limit = (i % 20 == 0)
        comments = fetch_pr_comments(pr_number, check_rate_limit=check_limit)
        ai_comments = filter_ai_comments(comments)

        # Extract concerns per reviewer
        for reviewer, rev_comments in ai_comments.items():
            if rev_comments:
                comment_counts[reviewer] += len(rev_comments)
                if args.verbose:
                    print(f"  {reviewer}: {len(rev_comments)} comments")

                concerns = extract_concerns_with_llm(
                    model, model_type, rev_comments, reviewer, pr_number
                )
                all_concerns.extend(concerns)
                if args.verbose and concerns:
                    print(f"    Extracted {len(concerns)} concerns")

    print()
    print(f"Total concerns extracted: {len(all_concerns)}")
    print()

    # Cluster similar concerns
    print("Clustering semantically similar concerns...")
    clusters = cluster_concerns_with_llm(model, model_type, all_concerns)
    print(f"Created {len(clusters)} concern clusters")

    # Compute statistics
    stats = compute_reviewer_stats(clusters, comment_counts)

    # Generate report
    report = generate_report(stats, clusters, len(pr_numbers), REPO)

    # Save data if requested
    if args.save_data:
        if args.anonymize_summaries and not args.anonymize:
            print("Warning: --anonymize-summaries requires --anonymize, ignoring.")
            args.anonymize_summaries = False

        print(f"\nExporting analysis data...")
        export_data = export_analysis_data(
            concerns=all_concerns,
            clusters=clusters,
            stats=stats,
            report=report,
            repo=REPO,
            prs_analyzed=len(pr_numbers),
            anonymize=args.anonymize,
            anonymize_summaries=args.anonymize_summaries,
            model=model if args.anonymize_summaries else None,
            model_type=model_type if args.anonymize_summaries else None,
        )

        with open(args.save_data, "w") as f:
            json.dump(export_data, f, indent=2)

        mode = "anonymized" if args.anonymize else "raw"
        print(f"Data saved to: {args.save_data} ({mode} mode)")

        # Also save markdown report
        report_path = args.save_data.rsplit(".", 1)[0] + ".md"
        if args.anonymize:
            # Generate anonymized version of report
            anon_report = generate_report(stats, clusters, len(pr_numbers), REPO, anonymize=True)
            with open(report_path, "w") as f:
                f.write(anon_report)
        else:
            with open(report_path, "w") as f:
                f.write(report)
        print(f"Report saved to: {report_path}")

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to: {args.output}")
    else:
        print()
        print("=" * 60)
        print()
        print(report)


if __name__ == "__main__":
    main()
