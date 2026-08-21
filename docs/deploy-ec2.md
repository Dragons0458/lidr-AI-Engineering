# Deploy to EC2 (kit only — **not executed**)

This guide has **not** been run against a real host. No EC2, no domain, no
TLS certificate has been issued from this repository. Follow it when (if)
that work is scheduled; until then treat every command as a checklist.

## Topology

```text
Internet ──:80/:443──► caddy ──► web:8501 (Streamlit, internal)
                              └──► (no published ports)
                         ai-service:8000 ──► postgres / redis
```

Three layers of the frontier, all required:

1. **Security group** — 22, 80, 443 from the internet. Nothing else.
2. **`ufw` on the VM** — same three ports (`deploy/bootstrap.sh`).
3. **`ports:` in Compose** — only `caddy` publishes. `web` uses `ports: !reset []`
   because Compose **merges** lists; without `!reset` `:8501` would leak.

## Prerequisites

- Ubuntu 24.04 on a `t3.medium` (4 GB RAM), disk ≥ 30 GB.
- Elastic IP.
- A DNS **A record** pointing at that IP **before** the first Caddy start.
  Let's Encrypt will not issue for `*.amazonaws.com`.
- GitHub packages namespace (`IMAGE_OWNER`) once images are pushed from `main`.

## Provision

```bash
ssh ubuntu@<host> 'bash -s' < deploy/bootstrap.sh
```

The script is idempotent: Docker Engine + Compose v2, 2G swap (HNSW builds),
`ufw`, unattended-upgrades without automatic reboot, `/opt/estimador-cag`.

## Ship the repo and the secrets

```bash
scp -r . ubuntu@<host>:/opt/estimador-cag/
scp .env ubuntu@<host>:/opt/estimador-cag/.env
ssh ubuntu@<host> 'chmod 600 /opt/estimador-cag/.env'
```

`.env` never travels through GitHub Actions. Required production keys:

- `APP_DOMAIN`, `IMAGE_OWNER`, `IMAGE_TAG`
- `CADDY_BASIC_AUTH_USER`, `CADDY_BASIC_AUTH_HASH` (`caddy hash-password`)
- `OPENAI_API_KEY`, `AI_SERVICE_TOKEN` / `ESTIMATE_API_KEY`, `RETRIEVAL_API_KEY`
- `POSTGRES_PASSWORD`

Keep `AI_SERVICE_TOKEN` and `ESTIMATE_API_KEY` **equal**. Compare by hash,
never `echo`:

```bash
printenv AI_SERVICE_TOKEN | sha256sum
printenv ESTIMATE_API_KEY | sha256sum
```

## Obtain the certificate and start

```bash
ssh ubuntu@<host>
cd /opt/estimador-cag
sudo cp deploy/estimator.service /etc/systemd/system/estimador-cag.service
sudo systemctl daemon-reload
sudo systemctl enable --now estimador-cag
```

Caddy requests the Let's Encrypt cert on first listen. If the A record was
late, stop, fix DNS, start again (rate limit: 5 certs per domain per week —
that is why `caddy_data` is a named volume).

## Restore the corpus

A fresh volume has no embeddings. Do **not** re-ingest through the API.

```bash
./scripts/restore_corpus.sh /opt/estimador-cag/backups/corpus.dump
```

Dump from a known-good local stack with `./scripts/dump_corpus.sh`.

## Verify the frontier

From your laptop, not from the VM:

```bash
python scripts/smoke_test_s15.py --base-url "https://$APP_DOMAIN" --skip-estimation
```

Expect: UI 200, `/_stcore/health` 200, model badge present, ports 8000 / 8501 /
5432 / 6379 refused, HTTP→HTTPS redirect, valid TLS handshake.

Deep checks from inside the network (after SSH):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec web \
  python scripts/smoke_test_s15.py \
    --base-url http://web:8501 \
    --ai-url http://ai-service:8000
```

## Daily operations

```bash
sudo journalctl -u estimador-cag -f
sudo systemctl restart estimador-cag
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs ai-service --tail=100
```

Safe restart ladder: `restart` → `--force-recreate` → rebuild image → `down`/`up`.
Never `down -v` unless you intend to drop the corpus.

## Rollback

Images are tagged with the git SHA. Point `IMAGE_TAG` at a previous digest
and recreate — do not rebuild:

```bash
# on the host
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=<previous-sha>/' .env
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --remove-orphans
```

## Common failures

| Symptom | Likely cause |
| --- | --- |
| Caddy loop of ACME errors | A record missing / pointing at the wrong IP |
| 401 on every call | `AI_SERVICE_TOKEN` ≠ `ESTIMATE_API_KEY` (fixture 5) |
| UI loads, API connection refused | `ESTIMATION_API_BASE_URL` still `localhost` (fixture 3) |
| `ai-service` dies on first boot | `depends_on` without health conditions (fixture 2) |
| Host can curl `:8000` | `ports:` leaked on `ai-service` (fixture 4) |
| AxiosError 403 on file upload | Streamlit XSRF vs proxy; prod command must keep `enableXsrfProtection=true` and a single origin |

## What is still missing for “real” production

- No redundancy (one VM).
- No automated backups (only `dump_corpus.sh` by hand).
- Deploy has a cutover (`up -d` recreates containers).
- Streamlit auth is Caddy basic_auth, not an IdP.
- `/health/ready` still uses sync clients inside an async endpoint
  (see [`scalability.md`](scalability.md)).
