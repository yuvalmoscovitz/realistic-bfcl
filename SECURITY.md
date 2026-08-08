# Security

## Secret Handling

Realistic-BFCL never requires credentials in the repository. Supply provider
keys through an explicit `--env-file`, `REALISTIC_BFCL_ENV_FILE`, or process
environment variables. Private `.env` variants are rejected by pre-commit;
only the explicitly named `.env.example` and `.env.template` templates are
allowed by filename, and gitleaks still scans their contents.

Install and run the hooks with:

```bash
pre-commit install
pre-commit run --all-files
```

The hooks block private `.env` files and run gitleaks against staged changes.
The gitleaks hook is pinned to `v8.30.1`. Do not add broad path allowlists for
benchmark artifacts: if synthetic text triggers a false positive, document and
scope the exception to the specific non-secret value.

## Full-History Audit

On 2026-08-08, gitleaks `8.30.1` scanned all 56 commits reachable with
`--log-opts=--all`, covering approximately 11.58 MB of Git history. It reported
no leaks, so no history rewrite was needed. Before the audit, a non-placeholder
fake GitHub PAT canary produced one finding and a nonzero exit, confirming that
the pinned scanner and default rules were active.

Re-run the audit from the repository root with:

```bash
gitleaks git . --log-opts=--all --redact=100
```

If a future audit finds a real credential, revoke it first and report the
affected commits. Do not rewrite shared history or force-push without explicit
maintainer approval.
