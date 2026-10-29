# Contributing

Thank you for your interest in RVG.

- **Issues** are welcome for bugs, unclear documentation and questions. Please report security
  problems privately (see [SECURITY.md](SECURITY.md)), not in a public issue.
- **Pull requests**: keep them focused, add or update tests, and run the checks below. Changes to
  the controller, the policy file (`src/rvg/resources/policies/universal.json`) or prompt resources
  change answers; they need a measured reason. The policy is one file for every knowledge base and
  language by design — per-KB or per-language thresholds, profiles or retrieval settings are not
  accepted.
- **Never commit** credentials, `.env` files, recorded tapes, private documents or evaluation labels.
  `scripts/check_dist.py` refuses such files in a built distribution.

```bash
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv build && uv run python scripts/check_dist.py dist/*.whl dist/*.tar.gz
```

Contributions are accepted under the Apache License 2.0 (see [LICENSE](LICENSE)).
Most design documents are in Korean; English issues and pull requests are fine.
