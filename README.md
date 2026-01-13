# AI Code Reviewer Analysis

LLM-as-judge analysis tool for comparing AI code reviewers. Determines which AI reviewers provide unique value vs. which ones just repeat what others already caught.

## What This Tool Does

1. **Fetches PR comments** from your GitHub repository
2. **Extracts discrete concerns** using an LLM (severity, category)
3. **Clusters similar concerns** to identify overlap across reviewers
4. **Generates a report** showing each reviewer's unique value

## Why This Matters

If you're using multiple AI code reviewers (Claude, Gemini, Cursor, Codex, CodeRabbit, etc.), you might be paying for redundant feedback. This tool answers:

- Which reviewers catch issues that others miss?
- Which reviewers have the highest signal-to-noise ratio?
- Are you getting value from each reviewer, or are they just repeating each other?

## Supported AI Reviewers

Out of the box, the tool recognizes:

- **Claude** (`claude[bot]`)
- **Gemini Code Assist** (`gemini-code-assist[bot]`)
- **Cursor** (`cursor[bot]`)
- **Codex/ChatGPT** (`chatgpt-codex-connector[bot]`)
- **CodeRabbit** (`coderabbitai[bot]`)
- **Sourcery** (`sourcery-ai[bot]`)
- **DeepSource** (`deepsource-autofix[bot]`)

Add your own by editing `AI_REVIEWERS` in the script.

## Installation

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency management
- [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated
- Google Cloud project with Vertex AI (recommended) OR Google AI API key

### Setup

```bash
git clone https://github.com/qfennessy/ai-code-reviewer-analysis.git
cd ai-code-reviewer-analysis

# Install dependencies with Poetry
poetry install
```

## Usage

### With Vertex AI (Recommended)

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

poetry run python scripts/analyze-ai-reviewers.py \
  --vertex \
  --project your-gcp-project \
  --repo owner/repo \
  --prs 50
```

### Options

```
--repo, -r              GitHub repository (owner/name) [required]
--prs, -n               Number of PRs to analyze (default: 20)
--output, -o            Output file for report (default: stdout)
--verbose, -v           Show detailed progress
--vertex                Use Vertex AI instead of Google AI Studio
--project, -p           GCP project ID for Vertex AI
--save-data FILE        Save extracted analysis data to JSON file
--anonymize             Anonymize data when saving (requires --save-data)
--anonymize-summaries   Use LLM to generalize concern summaries (slower)
```

### Examples

```bash
# Quick analysis of 20 PRs
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo facebook/react --prs 20

# Comprehensive analysis saved to file
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo your-org/your-repo \
  --prs 200 \
  --output analysis-report.md

# Verbose mode to see what's happening
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo your-org/your-repo \
  --prs 50 \
  --verbose

# Save raw analysis data for further processing
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo your-org/your-repo \
  --prs 50 \
  --save-data analysis.json

# Save anonymized data for sharing (strips identifiers)
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo your-org/your-repo \
  --prs 50 \
  --save-data analysis-anon.json \
  --anonymize

# Anonymize with LLM-generalized summaries (most thorough)
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo your-org/your-repo \
  --prs 50 \
  --save-data analysis-anon.json \
  --anonymize \
  --anonymize-summaries
```

## Sample Output

```markdown
# LLM-as-Judge Analysis Results

**Repository**: your-org/your-repo
**Date**: 2025-01-15
**PRs Analyzed**: 100
**Total Concerns Extracted**: 450
**Unique Concern Clusters**: 380

## Summary by Reviewer

| Reviewer | Comments | Concerns | Unique Catches | Unique Rate |
|----------|----------|----------|----------------|-------------|
| **Claude** | 142 | 180 | 127 | 70.6% |
| **Gemini** | 200 | 150 | 98 | 65.3% |
| **Cursor** | 180 | 120 | 74 | 61.7% |

## Recommendations

1. **Highest unique catch count**: Claude (127 unique concerns)
2. **Highest signal-to-noise ratio**: Claude (127/180 concerns are unique)
```

## How It Works

### Concern Extraction

For each AI reviewer comment, the LLM extracts:
- **Summary**: One-sentence description of the concern
- **Severity**: critical, high, medium, low
- **Category**: bug, security, performance, type_safety, error_handling, etc.

### Semantic Clustering

The tool clusters concerns **per-PR** to answer: "Did multiple reviewers catch the same issue?"

- Two concerns about "missing null check in getUserEmail()" get clustered together
- A concern about "add try/catch" and "handle exceptions" for the same code get clustered
- Different issues remain separate

### Unique Value Calculation

A concern is "unique" if only one reviewer raised it on that PR. The **unique rate** shows what percentage of a reviewer's concerns provide differentiated value.

## Data Preservation

Use `--save-data` to export the full analysis for further processing or sharing. Both a JSON data file and a markdown report are generated.

### Raw Mode

Saves all extracted data including original text snippets, file paths, and PR numbers:

```bash
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo owner/repo --prs 50 --save-data analysis.json

# Creates:
#   analysis.json - Full data with all identifiers
#   analysis.md   - Markdown report
```

### Anonymized Mode

For sharing analysis results without revealing codebase details:

```bash
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo owner/repo --prs 50 \
  --save-data analysis.json --anonymize

# Creates:
#   analysis.json - Anonymized data
#   analysis.md   - Markdown report (with "anonymized-repository")
```

Anonymization applies:
- Repository name → `anonymized-repository`
- PR numbers → sequential IDs (`PR-001`, `PR-002`, ...)
- File paths → hashed with extension preserved (`file-001.ts`)
- Line numbers → removed
- Original text snippets → removed
- Statistics and severity/category distributions → preserved

### LLM Summary Generalization

For thorough anonymization, add `--anonymize-summaries` to strip code-specific details from concern summaries using the LLM:

```bash
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo owner/repo --prs 50 \
  --save-data analysis.json --anonymize --anonymize-summaries
```

This transforms summaries like "Missing null check in getUserEmail()" to "Missing null check in function return value".

## Rate Limiting

The tool includes automatic rate limiting to avoid hitting GitHub's API limits:

- 100ms minimum delay between requests
- Automatic detection of rate limit status
- Exponential backoff when limits are approached

For large analyses (500+ PRs), consider running in batches:

```bash
# Run 200 PRs at a time
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo owner/repo --prs 200 --output batch1.md
# Wait 1 hour
poetry run python scripts/analyze-ai-reviewers.py \
  --vertex --project your-gcp-project \
  --repo owner/repo --prs 200 --output batch2.md
```

## Cost

- **LLM costs**: Uses Gemini 2.0 Flash, which is very cost-effective
- **GitHub API**: Free within rate limits (5000 requests/hour for authenticated users)

Typical cost for analyzing 100 PRs: ~$0.10-0.50 depending on comment volume.

## Adding Custom Reviewers

Edit `AI_REVIEWERS` in the script:

```python
AI_REVIEWERS = {
    "claude": ["claude[bot]", "@claude"],
    "gemini": ["gemini-code-assist[bot]"],
    # Add your custom reviewer:
    "my-bot": ["my-custom-bot[bot]"],
}
```

## License

MIT

## Author

Created by [Quentin](https://github.com/qfennessy) based on analysis of AI code reviewers in production.

## Related

- [Blog post about AI code reviewer ROI](https://example.com) (coming soon)
- [Original analysis that inspired this tool](docs/sample-analysis.md)
