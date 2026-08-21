# Security Policy

ExpoChat is a supervisor that executes AI-planned changes against real workspaces. It is
built to run on a trusted host behind authentication. Please treat it accordingly.

## Reporting a vulnerability

**Do not open a public issue for security problems.** Instead, report privately:

- Use GitHub's **Report a vulnerability** (Security → Advisories) on this repository, or
- Email the maintainers at **hallo@expoxr.com** with details and reproduction steps.

Please include the affected version/commit, impact, and a proof of concept if possible.
We aim to acknowledge reports within a few business days.

## Security model (what protects you)

- **Authentication.** Admin login is required. Passwords use Argon2id
  (`ADMIN_PASSWORD_HASH`); the app refuses to start with the placeholder/default password
  unless `ALLOW_INSECURE_PASSWORD=true` (a dev-only escape hatch).
- **Session + CSRF.** Signed session cookies (`SESSION_SECRET`); mutations require a CSRF
  token. Use HTTPS with `SECURE_COOKIE=true` and explicit `ALLOWED_ORIGINS` in production.
- **Credential encryption at rest.** Cloud provider API keys are Fernet-encrypted with
  `CREDENTIAL_ENCRYPTION_KEY` before storage.
- **Service isolation.** The Brain and Worker run as separate, credential-isolated
  containers (`read_only`, `cap_drop: ALL`, `no-new-privileges`) on an internal network.
  Workers reach Ollama only through a token-gated proxy with a fixed route allow-list.
- **Path sandboxing.** All workspace file access is confined to `ALLOWED_ROOTS`; destructive
  file operations snapshot first and runs apply only behind a verified snapshot.

## Deployment hardening checklist

- Set strong, unique values for `ADMIN_PASSWORD_HASH`, `SESSION_SECRET`,
  `CREDENTIAL_ENCRYPTION_KEY`, and `WORKER_TOKEN`. Rotate anything ever copied from an
  example.
- Serve over HTTPS; set `SECURE_COOKIE=true`, `ALLOWED_ORIGINS`, and `FORWARDED_ALLOW_IPS`
  to your actual reverse-proxy IP.
- Keep `.env` at mode `600` and never commit it (`.gitignore` already excludes it).
- Restrict `ALLOWED_ROOTS` and host mounts to exactly what the app needs.

## Supported versions

This project is pre-1.0; security fixes target the latest default branch.
