"""Tests for the AI code reviewer analysis tool."""

import importlib.util
import sys
from pathlib import Path

# Load the script module with hyphens in name
script_path = Path(__file__).parent.parent / "scripts" / "analyze-ai-reviewers.py"
spec = importlib.util.spec_from_file_location("analyze_ai_reviewers", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules["analyze_ai_reviewers"] = module
spec.loader.exec_module(module)

# Import from loaded module
AI_REVIEWERS = module.AI_REVIEWERS
Anonymizer = module.Anonymizer
Category = module.Category
Concern = module.Concern
ConcernCluster = module.ConcernCluster
ReviewerStats = module.ReviewerStats
Severity = module.Severity
cluster_to_dict = module.cluster_to_dict
concern_to_dict = module.concern_to_dict
filter_ai_comments = module.filter_ai_comments
identify_reviewer = module.identify_reviewer
stats_to_dict = module.stats_to_dict


class TestIdentifyReviewer:
    """Tests for the identify_reviewer function."""

    def test_identifies_claude_bot(self):
        assert identify_reviewer("claude[bot]") == "claude"

    def test_identifies_claude_mention(self):
        assert identify_reviewer("@claude") == "claude"

    def test_identifies_gemini_bot(self):
        assert identify_reviewer("gemini-code-assist[bot]") == "gemini"

    def test_identifies_cursor_bot(self):
        assert identify_reviewer("cursor[bot]") == "cursor"

    def test_identifies_codex_bot(self):
        assert identify_reviewer("chatgpt-codex-connector[bot]") == "codex"

    def test_identifies_coderabbit_bot(self):
        assert identify_reviewer("coderabbitai[bot]") == "coderabbit"

    def test_case_insensitive(self):
        assert identify_reviewer("CLAUDE[BOT]") == "claude"
        assert identify_reviewer("Gemini-Code-Assist[bot]") == "gemini"

    def test_returns_none_for_human(self):
        assert identify_reviewer("octocat") is None
        assert identify_reviewer("human-reviewer") is None

    def test_returns_none_for_empty(self):
        assert identify_reviewer("") is None


class TestAnonymizer:
    """Tests for the Anonymizer class."""

    def test_anonymize_pr_sequential(self):
        anon = Anonymizer()
        assert anon.anonymize_pr(123) == "PR-001"
        assert anon.anonymize_pr(456) == "PR-002"
        assert anon.anonymize_pr(789) == "PR-003"

    def test_anonymize_pr_deterministic(self):
        anon = Anonymizer()
        assert anon.anonymize_pr(123) == "PR-001"
        assert anon.anonymize_pr(123) == "PR-001"  # Same input = same output

    def test_anonymize_file_preserves_extension(self):
        anon = Anonymizer()
        result = anon.anonymize_file("src/auth/validator.ts")
        assert result.endswith(".ts")
        assert result.startswith("file-")

    def test_anonymize_file_no_extension(self):
        anon = Anonymizer()
        result = anon.anonymize_file("Makefile")
        assert result == "file-001"

    def test_anonymize_file_deterministic(self):
        anon = Anonymizer()
        first = anon.anonymize_file("src/index.js")
        second = anon.anonymize_file("src/index.js")
        assert first == second

    def test_anonymize_file_none(self):
        anon = Anonymizer()
        assert anon.anonymize_file(None) is None

    def test_anonymize_concern(self):
        anon = Anonymizer()
        concern = {
            "summary": "Missing null check",
            "severity": "high",
            "category": "null_safety",
            "reviewer": "claude",
            "pr_number": 123,
            "original_text": "The variable x might be null",
            "file_path": "src/utils.ts",
            "line_number": 42,
        }
        result = anon.anonymize_concern(concern)

        assert result["summary"] == "Missing null check"  # Preserved
        assert result["severity"] == "high"  # Preserved
        assert result["category"] == "null_safety"  # Preserved
        assert result["reviewer"] == "claude"  # Preserved
        assert result["pr_number"] == "PR-001"  # Anonymized
        assert result["original_text"] is None  # Removed
        assert result["file_path"].endswith(".ts")  # Anonymized
        assert result["line_number"] is None  # Removed


class TestSerializationHelpers:
    """Tests for serialization helper functions."""

    def test_concern_to_dict(self):
        concern = Concern(
            summary="Test concern",
            severity=Severity.HIGH,
            category=Category.BUG,
            reviewer="claude",
            pr_number=123,
            original_text="Original text",
            file_path="src/test.py",
            line_number=10,
        )
        result = concern_to_dict(concern)

        assert result["summary"] == "Test concern"
        assert result["severity"] == "high"
        assert result["category"] == "bug"
        assert result["reviewer"] == "claude"
        assert result["pr_number"] == 123
        assert result["original_text"] == "Original text"
        assert result["file_path"] == "src/test.py"
        assert result["line_number"] == 10

    def test_cluster_to_dict(self):
        concern = Concern(
            summary="Test concern",
            severity=Severity.MEDIUM,
            category=Category.PERFORMANCE,
            reviewer="gemini",
            pr_number=456,
            original_text="Text",
        )
        cluster = ConcernCluster(
            representative_summary="Representative summary",
            concerns=[concern],
            reviewers={"gemini", "claude"},
        )
        result = cluster_to_dict(cluster)

        assert result["representative_summary"] == "Representative summary"
        assert len(result["concerns"]) == 1
        assert result["reviewers"] == ["claude", "gemini"]  # Sorted
        assert result["is_unique"] is False

    def test_cluster_to_dict_unique(self):
        concern = Concern(
            summary="Unique concern",
            severity=Severity.LOW,
            category=Category.CODE_STYLE,
            reviewer="cursor",
            pr_number=789,
            original_text="Text",
        )
        cluster = ConcernCluster(
            representative_summary="Unique summary",
            concerns=[concern],
            reviewers={"cursor"},
        )
        result = cluster_to_dict(cluster)

        assert result["is_unique"] is True

    def test_stats_to_dict(self):
        stats = ReviewerStats(
            name="claude",
            total_comments=10,
            total_concerns=8,
            unique_concerns=5,
            concerns_by_severity={"critical": 1, "high": 2},
            concerns_by_category={"bug": 3},
            unique_catches=["Catch 1", "Catch 2"],
        )
        result = stats_to_dict(stats)

        assert result["name"] == "claude"
        assert result["total_comments"] == 10
        assert result["total_concerns"] == 8
        assert result["unique_concerns"] == 5
        assert result["concerns_by_severity"] == {"critical": 1, "high": 2}
        assert result["concerns_by_category"] == {"bug": 3}
        assert result["unique_catches"] == ["Catch 1", "Catch 2"]


class TestFilterAIComments:
    """Tests for the filter_ai_comments function."""

    def test_filters_claude_comments(self):
        comments = [
            {"user": {"login": "claude[bot]"}, "body": "Review comment"},
            {"user": {"login": "octocat"}, "body": "Human comment"},
        ]
        result = filter_ai_comments(comments)

        assert len(result["claude"]) == 1
        assert result["claude"][0]["body"] == "Review comment"

    def test_filters_multiple_reviewers(self):
        comments = [
            {"user": {"login": "claude[bot]"}, "body": "Claude comment"},
            {"user": {"login": "gemini-code-assist[bot]"}, "body": "Gemini comment"},
            {"user": {"login": "cursor[bot]"}, "body": "Cursor comment"},
        ]
        result = filter_ai_comments(comments)

        assert len(result["claude"]) == 1
        assert len(result["gemini"]) == 1
        assert len(result["cursor"]) == 1

    def test_handles_null_user(self):
        """Test handling of comments with null user (deleted/ghost users)."""
        comments = [
            {"user": None, "body": "Ghost comment"},
            {"user": {"login": "claude[bot]"}, "body": "Valid comment"},
        ]
        result = filter_ai_comments(comments)

        assert len(result["claude"]) == 1
        # Should not raise an error

    def test_handles_missing_user(self):
        """Test handling of comments with missing user field."""
        comments = [
            {"body": "Comment without user"},
            {"user": {"login": "claude[bot]"}, "body": "Valid comment"},
        ]
        result = filter_ai_comments(comments)

        assert len(result["claude"]) == 1

    def test_empty_comments(self):
        result = filter_ai_comments([])

        for reviewer in AI_REVIEWERS.keys():
            assert result[reviewer] == []


class TestConcernCluster:
    """Tests for ConcernCluster dataclass properties."""

    def test_is_unique_single_reviewer(self):
        cluster = ConcernCluster(
            representative_summary="Test",
            reviewers={"claude"},
        )
        assert cluster.is_unique is True

    def test_is_unique_multiple_reviewers(self):
        cluster = ConcernCluster(
            representative_summary="Test",
            reviewers={"claude", "gemini"},
        )
        assert cluster.is_unique is False

    def test_unique_reviewer_single(self):
        cluster = ConcernCluster(
            representative_summary="Test",
            reviewers={"cursor"},
        )
        assert cluster.unique_reviewer == "cursor"

    def test_unique_reviewer_multiple(self):
        cluster = ConcernCluster(
            representative_summary="Test",
            reviewers={"claude", "gemini"},
        )
        assert cluster.unique_reviewer is None


class TestSeverityAndCategory:
    """Tests for Severity and Category enums."""

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"

    def test_category_values(self):
        assert Category.BUG.value == "bug"
        assert Category.SECURITY.value == "security"
        assert Category.PERFORMANCE.value == "performance"
        assert Category.NULL_SAFETY.value == "null_safety"
