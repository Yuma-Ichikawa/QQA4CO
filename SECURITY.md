# Security policy

## Supported versions

Only the latest released version of QQA4CO on PyPI receives security
patches. Older releases are unsupported.

| Version | Supported |
|---|---|
| `>= 0.3.0` | yes |
| `< 0.3.0` | no — please upgrade |

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security problems.
Instead, use GitHub's private security reporting feature:

<https://github.com/Yuma-Ichikawa/QQA4CO/security/advisories/new>

Include:

1. A clear description of the issue.
2. A minimal reproducer (Python / shell script).
3. The affected version (`qqa.__version__`) and Python / PyTorch versions.
4. (Optional) a suggested patch.

We aim to acknowledge a report within 7 days, and to publish a fix
within 30 days when feasible. We will credit reporters in the release
notes unless they request anonymity.

## Out of scope

QQA4CO is a research-grade scientific computing library. Reports
about pickle deserialisation of attacker-controlled `AnnealResult`
files, or about resource exhaustion when the user passes deliberately
adversarial problem sizes, are *expected* behaviour and out of scope.
