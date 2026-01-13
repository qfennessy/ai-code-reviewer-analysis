# AI Code Reviewer Analysis

**Date Created**: 2026-01-10
**Date Last Updated**: 2026-01-13

Analysis of AI code review tools used on this repository over the last 60 days (200 PRs).

> **Note**: This document now includes both heuristic-based analysis (keyword matching) and
> LLM-as-judge semantic analysis. See [LLM-as-Judge Analysis](#llm-as-judge-analysis) for
> the deeper semantic analysis that identifies truly unique insights vs. rephrased duplicates.

## Overview

Four AI code review tools are active on this repository:
- **Claude** (`claude[bot]`) - Anthropic's Claude via GitHub Actions
- **Gemini Code Assist** (`gemini-code-assist[bot]`) - Google's Gemini
- **Cursor** (`cursor[bot]`) - Cursor AI
- **Codex** (`chatgpt-codex-connector[bot]`) - OpenAI's Codex

## Volume Analysis

### Total Comments (Last 60 Days, 200 PRs)

| Reviewer | PR Comments | Inline Reviews | Total |
|----------|-------------|----------------|-------|
| **Gemini Code Assist** | 226 | 490 | **716** |
| **Cursor** | 0 | 483 | **483** |
| **Claude** | 285 | 0 | **285** |
| **Codex** | 13 | 154 | **167** |
| **TOTAL** | 524 | 1,127 | **1,651** |

### Feedback Style

| Reviewer | Style | Description |
|----------|-------|-------------|
| **Gemini** | Balanced | Both PR-level summaries AND inline code suggestions |
| **Cursor** | Inline-only | No general PR discussion, just code-level feedback |
| **Claude** | Summary-only | PR reviews/summaries but no inline suggestions |
| **Codex** | Both | Lower volume but covers both types |

## Unique Value Analysis

### Exclusive Catches

Issues found by ONLY that reviewer (not caught by others reviewing the same PR):

| Reviewer | Exclusive Catches | Top Strengths |
|----------|------------------|---------------|
| **Claude** | 41 | error_handling, security, performance |
| **Gemini** | 33 | type_safety, race_condition, performance |
| **Cursor** | 6 | null_safety, security, testing |
| **Codex** | 6 | race_condition, null_safety, security |

### Concern Coverage by Reviewer

Based on keyword/pattern analysis of 50 PRs with 2+ AI reviewers:

| Concern Type | Claude | Gemini | Cursor | Codex |
|--------------|--------|--------|--------|-------|
| type_safety | 40 | **50** | 23 | 21 |
| error_handling | **37** | 21 | 16 | 7 |
| security | **31** | 25 | 15 | 6 |
| performance | **42** | 39 | 15 | 6 |
| testing | **46** | 45 | 23 | 7 |
| logic_bug | 43 | **50** | 32 | 22 |
| null_safety | 24 | **29** | 21 | 10 |
| race_condition | 40 | **41** | 18 | 13 |
| code_style | 38 | **42** | 24 | 14 |
| documentation | 44 | **50** | 24 | 9 |

**Bold** indicates highest coverage for that concern type.

## Actionability Analysis

Measured by: inline comments that were followed by commits within 2 hours (suggesting the feedback was acted upon).

| Reviewer | Inline Comments | Led to Commits | Actionability Rate |
|----------|-----------------|----------------|-------------------|
| **Codex** | 32 | 30 | **93.8%** |
| **Gemini** | 108 | 96 | **88.9%** |
| **Cursor** | 102 | 67 | 65.7% |
| Claude | 0 (summary only) | N/A | N/A |

## Reviewer Profiles

### Claude
- **Role**: High-level architectural review
- **Strengths**: Error handling, security, performance concerns
- **Style**: PR-level summaries only, no inline code suggestions
- **Value**: Best "second opinion" on approach and architecture
- **Exclusive catches**: 41 (highest)

### Gemini Code Assist
- **Role**: Primary all-around reviewer
- **Strengths**: Type safety, documentation, logic bugs
- **Style**: Both summaries AND inline suggestions
- **Value**: High volume (716 comments) with high actionability (89%)
- **Exclusive catches**: 33

### Cursor
- **Role**: Focused inline code reviewer
- **Strengths**: Null safety, security edge cases
- **Style**: Inline comments only, no PR summaries
- **Value**: Good for catching null/undefined issues
- **Concern**: Lower actionability (66%) suggests higher noise
- **Exclusive catches**: 6

### Codex
- **Role**: High-signal, low-volume reviewer
- **Strengths**: Race conditions, null safety
- **Style**: Mostly inline, few general comments
- **Value**: Highest actionability (94%) - when it speaks, listen
- **Exclusive catches**: 6

## Recommendations

### Keep All Four

Each reviewer catches different things:
1. **Claude** for architectural review and security concerns
2. **Gemini** as primary reviewer (best volume + quality balance)
3. **Codex** for high-signal catches (highest actionability)
4. **Cursor** for null safety (consider ROI given 66% actionability)

### Potential Optimization

Consider whether Cursor's volume (483 inline comments) justifies its lower actionability rate (66%). If ~160 comments are noise, that's developer time spent reviewing unhelpful suggestions.

## LLM-as-Judge Analysis

The heuristic analysis above uses keyword matching, which may miss semantically similar
concerns or misclassify issues. This section presents results from an LLM-based semantic
analysis that extracts discrete concerns and clusters similar issues across reviewers.

### Methodology

1. **Concern Extraction**: For each AI reviewer comment, an LLM extracts discrete,
   actionable concerns with severity (critical/high/medium/low) and category classifications.
2. **Semantic Clustering**: Similar concerns across reviewers are grouped to identify
   which issues were caught by multiple reviewers vs. uniquely by one.
3. **Unique Value Assessment**: Concerns raised by only one reviewer are flagged as
   "unique catches" - the true measure of each reviewer's differentiated value.

### LLM-as-Judge Results (100 PRs)

| Reviewer | Comments | Concerns | Critical | High | Severity Focus |
|----------|----------|----------|----------|------|----------------|
| **Claude** | 125 | 343 | 16 | 14 | 8.7% |
| **Gemini** | 374 | 275 | 15 | 58 | 26.5% |
| **Cursor** | 290 | 278 | 3 | 56 | 21.2% |
| **Codex** | 86 | 80 | 20 | 53 | **91.3%** |

**Key Findings**:
- **976 total concerns** extracted from 100 PRs
- **Codex** has the highest severity focus - 91.3% of its concerns are critical or high severity
- **Claude** generates the most concerns per comment (2.74 concerns/comment)
- **Gemini** has the best volume/coverage balance (374 comments, 275 concerns)

### Concern Severity Distribution (100 PRs)

| Reviewer | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| **Claude** | 16 | 14 | 136 | 177 |
| **Gemini** | 15 | 58 | 191 | 11 |
| **Cursor** | 3 | 56 | 166 | 53 |
| **Codex** | 20 | 53 | 7 | 0 |

**Key Finding**: Codex has the highest severity focus - 91.3% of its concerns are critical
or high severity. When Codex speaks, it's almost always about something important.

### Concern Category Distribution (100 PRs)

| Reviewer | Bug | Security | Performance | Type Safety | Error Handling |
|----------|-----|----------|-------------|-------------|----------------|
| **Claude** | 36 | 10 | 41 | 10 | 35 |
| **Gemini** | 61 | 2 | 18 | 13 | 20 |
| **Cursor** | 167 | 7 | 20 | 5 | 25 |
| **Codex** | 68 | 4 | 3 | 1 | 2 |

**Specializations**:
- **Claude**: Performance optimization, testing gaps, architectural concerns
- **Gemini**: Bug detection, testing errors, documentation issues
- **Cursor**: UI/UX bugs, state management, accessibility
- **Codex**: Data integrity bugs, state management, race conditions

### Notable Unique Catches

#### Claude's Unique Strengths
- Race condition where quality signals are invalidated before backend completion
- Missing test coverage for error scenarios and performance implications
- Frontend tests lack basic accessibility validation
- Function uses brittle string matching, should use enum

#### Gemini's Unique Strengths
- Mock data uses incorrect property names, leading to test failures
- Test should check if function is callable before invoking
- Data collection period contradicts analysis period in documentation
- Unused imports that should be removed

#### Cursor's Unique Strengths
- Auto-dismiss timer resets on every parent re-render
- Invalid nested interactive HTML (accessibility violation)
- Component collapses unexpectedly during active extraction
- Version history list does not refresh after restore operation

#### Codex's Unique Strengths
- Batches collapse immediately, defeating the 'start expanded' behavior
- Merge drawer state is not cleared, leading to stale state
- Timeline validator reads relationship data from wrong Firestore collection
- Key emitted but not declared in schema, causing runtime error

### Overlapping Concerns

21 concerns (9.5%) were raised by multiple reviewers, indicating redundant coverage in these areas:

- Timestamp-based ID generation robustness (Claude + Gemini)
- Duplicated functions that should be factored out (Claude + Gemini)
- Magic strings that should be constants (Claude + Gemini)
- Restore function validation for soft-deleted entities (Codex + Cursor)
- Conflicting field paths in Firestore updates (Codex + Cursor)

### Extended Analysis (400 PRs - 2026-01-13)

An extended analysis covering approximately 400 PRs was conducted to validate findings with
a larger dataset. Note: GitHub API rate limits prevented analysis of all 800 targeted PRs,
but 400 PRs provides statistically significant results.

#### Summary Statistics (400 PRs)

| Reviewer | Comments | Concerns | Unique Catches | Unique Rate |
|----------|----------|----------|----------------|-------------|
| **Claude** | 142 | 392 | 277 | **70.7%** |
| **Gemini** | 452 | 331 | 217 | 65.6% |
| **Cursor** | 322 | 317 | 195 | 61.5% |
| **Codex** | 102 | 96 | 36 | 37.5% |

**Key Findings (400 PRs)**:
- **1136 total concerns** extracted from 400 PRs
- **852 unique concern clusters** after semantic deduplication
- **127 overlapping concerns** (11.2%) were caught by multiple reviewers
- Claude has the highest unique rate (70.7%), meaning 70% of its concerns are not duplicated by others

#### Severity Distribution (400 PRs)

| Reviewer | Critical | High | Medium | Low | Critical+High % |
|----------|----------|------|--------|-----|-----------------|
| **Claude** | 12 | 13 | 156 | 211 | 6.4% |
| **Gemini** | 19 | 72 | 224 | 16 | 27.5% |
| **Cursor** | 4 | 63 | 190 | 60 | 21.1% |
| **Codex** | 23 | 64 | 9 | 0 | **90.6%** |

**Codex's severity focus confirmed**: With 90.6% of concerns being critical or high severity,
Codex continues to be the highest-signal reviewer. When Codex raises a concern, it's almost
always about something important.

#### Category Distribution (400 PRs)

| Reviewer | Bug | Security | Performance | Type Safety | Error Handling |
|----------|-----|----------|-------------|-------------|----------------|
| **Claude** | 34 | 12 | 54 | 14 | 39 |
| **Gemini** | 70 | 2 | 21 | 16 | 23 |
| **Cursor** | 197 | 8 | 18 | 6 | 27 |
| **Codex** | 83 | 5 | 4 | 1 | 1 |

**Reviewer Specializations (confirmed)**:
- **Claude**: Performance optimization (54), error handling (39), documentation guidance
- **Gemini**: Bug detection (70), type safety (16), test quality
- **Cursor**: Bug detection (197), focuses heavily on UI/UX and state management
- **Codex**: Bug detection (83), data integrity, state management

#### Notable Unique Catches (400 PRs)

**Claude (277 unique concerns)**:
- Documentation guidance on test execution optimization
- Architecture recommendations for derived relationship calculations
- Validation rules for complex permutation types
- Test fixture generation from specifications

**Gemini (217 unique concerns)**:
- Command-line argument parsing using proper libraries (yargs)
- Empty catch block hiding debugging information
- Type validation in list response unwrapping
- Inconsistent permutation counts in documentation

**Cursor (195 unique concerns)**:
- Dry-run mode providing misleading output
- Invalid parameter causing silent behavior changes
- Unused function parameters creating potential bugs
- Deduplication only checking open issues

**Codex (36 unique concerns)**:
- Entity resolution not fully resolving duplicates
- Incompatible metadata format causing analytics data loss
- Duplicate mention_id values when merging chunks
- Migration path issues causing 404 errors

#### Top Overlapping Concerns (400 PRs)

Concerns raised by multiple reviewers indicate areas of consensus:

1. **Hardcoded repository name** (All 4 reviewers) - Non-portable scripts
2. **Shell escaping incomplete** (Codex + Cursor) - Security concern
3. **Missing success/error handlers** (Cursor + Gemini) - UX regression
4. **Incorrect filtering logic** (Claude + Codex + Cursor) - Data integrity
5. **Legacy session migration** (Codex + Cursor) - Data loss risk

### Updated Recommendations

Based on the LLM-as-judge analysis, the recommendations differ from the heuristic analysis:

#### Keep All Four Reviewers (Confirmed)

Each reviewer catches significant unique issues (updated with 400-PR data):

1. **Claude** (70.7% unique rate): Best for architectural review, performance concerns,
   and testing strategy. Highest unique rate means most of its feedback is differentiated.

2. **Gemini** (65.6% unique rate): Best all-around reviewer with highest comment volume (452)
   and strong bug detection. Good balance of coverage and unique insights.

3. **Cursor** (61.5% unique rate): Strong on bug detection (197 bug concerns), especially
   UI/UX issues and state management problems that others miss.

4. **Codex** (37.5% unique rate, **90.6% critical+high**): Lower unique rate but highest severity
   focus. When Codex flags something, it's almost always about something important.

#### ROI Assessment (400 PRs)

| Reviewer | Concerns | Critical/High | Focus Area | Developer Time ROI |
|----------|----------|---------------|------------|-------------------|
| Claude | 392 | 25 (6.4%) | Architecture, performance | HIGH - Highest unique rate |
| Gemini | 331 | 91 (27.5%) | Bugs, testing | HIGH - Best volume/quality |
| Cursor | 317 | 67 (21.1%) | UI/UX bugs | HIGH - Catches bugs others miss |
| Codex | 96 | 87 (90.6%) | Critical issues | **HIGHEST** - Almost all critical/high |

**Key insight**: Codex has the highest signal quality - when it comments, 90.6% of concerns
are critical or high severity. It's the "oracle" reviewer for serious issues.

**Note on unique rates**: Codex's lower unique rate (37.5%) reflects that other reviewers
often catch the same critical issues, but Codex consistently identifies the most severe
problems first or with more precision.

### Running the Analysis

To reproduce or update this analysis:

```bash
# From apps/interviewer directory (uses Poetry environment)
cd apps/interviewer

# Run with Vertex AI (uses Application Default Credentials)
poetry run python ../../scripts/analyze-ai-reviewers.py --vertex --prs 20

# Or with Google AI API key
GOOGLE_API_KEY=your-key poetry run python ../../scripts/analyze-ai-reviewers.py --prs 20

# Save output to file
poetry run python ../../scripts/analyze-ai-reviewers.py --vertex --prs 50 --output results.md
```

See `scripts/analyze-ai-reviewers.py` for implementation details.

## Methodology

### Data Collection
- **Initial period**: 2025-11-11 to 2026-01-10 (approximately 60 days)
- **Extended analysis**: 2026-01-13 (400 PRs covering broader history)
- **PRs analyzed**: 100 PRs (initial), 400 PRs (extended)
- **Sources**: GitHub API for comments and commits
- **Rate limiting**: Script includes automatic rate limit detection and backoff

### Concern Detection
Pattern-based keyword matching for concern types:
- `security`: security, vulnerab, injection, xss, csrf, auth, sanitiz, credential
- `performance`: performance, slow, optimi, efficien, cache, memo, complexity
- `null_safety`: null, undefined, optional, None, TypeError, KeyError
- `error_handling`: error handl, exception, try/catch, throw, raise
- `type_safety`: type, typing, TypeScript, any, interface, Pydantic
- `race_condition`: race, concurren, async, await, Promise, deadlock
- `code_style`: naming, convention, readab, refactor, duplicate, DRY
- `testing`: test, coverage, mock, assert, unit test, edge case
- `logic_bug`: bug, logic, incorrect, wrong, should be, off-by-one

### Actionability
Defined as: inline comment followed by a commit within 2 hours on the same PR.

## References

- **LLM-as-judge script**: `scripts/analyze-ai-reviewers.py` (primary analysis tool)
- Raw data available via GitHub API
- Data collection: `gh api` commands for PR comments and reviews
- Related issue: [#2032](https://github.com/sagacious-heritage/cocos-story/issues/2032)

Note: The original heuristic analysis scripts (keyword matching) were temporary and have been
superseded by the LLM-as-judge semantic analysis which provides more accurate results.
