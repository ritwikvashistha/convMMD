# Security policy

`convMMD` is a research preview rather than a production inference service.
Security reports are still welcome, especially for checkpoint loading,
dependency behavior, unsafe file handling, or examples that could disclose
user data.

## Supported versions

Security fixes are considered for the current 0.2.x public release line. Older
development snapshots are not supported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Email
`ritwikvashistha@gmail.com` with the subject `convMMD security report` and
include:

- the affected version or commit;
- a concise description of the impact;
- steps or a minimal example that reproduce the issue;
- any suggested mitigation; and
- whether the report may be acknowledged publicly after a fix.

GitHub private vulnerability reporting may also be used once it is enabled for
the public repository. Please avoid including unrelated personal, proprietary,
or sensitive data. Reports will be reviewed privately, but this research
project does not promise a fixed response or remediation time.

For ordinary bugs or scientific questions with no security impact, use the
public issue tracker.

## Checkpoint trust

Load convMMD checkpoints only from a source you trust. The package uses
`torch.load(..., weights_only=True)` and validates the payload schema, tensor
metadata, and model configuration, but those checks occur during and after
PyTorch deserialization. They do not make loading an attacker-controlled file
a supported security boundary. Do not expose checkpoint loading directly to
untrusted uploads or network input.
