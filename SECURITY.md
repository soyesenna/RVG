# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **"Report a vulnerability"** (Security
Advisories) on https://github.com/soyesenna/RVG, or by email to Jooyoung Kim
(kjy915875@gmail.com) — not in a public issue. Include the affected version, a description and, if
possible, a minimal reproduction. We will acknowledge the report and coordinate a fix and
disclosure.

## What the package keeps out

- Private documents, evaluation labels, recorded tapes and credentials are never part of this
  repository or its distributions (`scripts/check_dist.py` checks every build).
- Recorded tapes contain document text and model inputs/outputs; keep them in protected storage.
  Authorization headers are removed before recording.
- Provider API keys come from environment variables (or `.env` for standalone use only); never put
  them in `.gitmodules`, lock files, configuration examples or prompts.
- The principal (who is asking) must come from a trusted authentication layer
  (`rvg.TrustedPrincipal`); never derive permissions from the question text.

Korean summary: [SECURITY.ko.md](SECURITY.ko.md).
