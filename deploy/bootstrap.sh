#!/usr/bin/env bash
# Idempotent host bootstrap for a single Ubuntu 24.04 VM (t3.medium, ≥30 GB disk).
# Usage: ssh ubuntu@<host> 'bash -s' < deploy/bootstrap.sh
set -euo pipefail

APP_USER="${APP_USER:-ubuntu}"
APP_DIR="${APP_DIR:-/opt/estimador-cag}"

export DEBIAN_FRONTEND=noninteractive

echo "==> Base packages"
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  postgresql-client \
  ufw \
  unattended-upgrades

echo "==> Docker Engine + Compose v2 plugin"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  # shellcheck disable=SC1091
  . /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y --no-install-recommends \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
fi

usermod -aG docker "${APP_USER}" || true

echo "==> Docker daemon log rotation"
if [[ ! -f /etc/docker/daemon.json ]]; then
  cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
  systemctl restart docker
fi

echo "==> 2G swap (HNSW index builds on 4 GB RAM)"
if ! swapon --show | grep -q '/swapfile'; then
  if [[ ! -f /swapfile ]]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Firewall: 22/80/443 only"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> Unattended upgrades (no automatic reboot)"
dpkg-reconfigure -f noninteractive unattended-upgrades
cat > /etc/apt/apt.conf.d/51-estimador-no-reboot <<'EOF'
Unattended-Upgrade::Automatic-Reboot "false";
EOF

echo "==> App directory"
mkdir -p "${APP_DIR}"
chown "${APP_USER}:${APP_USER}" "${APP_DIR}"

if [[ -f "${APP_DIR}/deploy/estimator.service" ]]; then
  echo "==> systemd unit"
  cp "${APP_DIR}/deploy/estimator.service" /etc/systemd/system/estimador-cag.service
  systemctl daemon-reload
  systemctl enable estimador-cag.service
fi

cat <<EOF

Bootstrap complete.

Next:
  1. Point the domain A record at this host *before* the first Caddy start
     (Let's Encrypt will not issue for *.amazonaws.com).
  2. scp the repo into ${APP_DIR} (or git clone).
  3. scp the secrets file to ${APP_DIR}/.env and chmod 600 it. Never commit it.
  4. Set APP_DOMAIN, IMAGE_OWNER, CADDY_BASIC_AUTH_USER, CADDY_BASIC_AUTH_HASH.
  5. systemctl start estimador-cag
  6. Restore the corpus: ./scripts/restore_corpus.sh /path/to/corpus.dump
  7. Run the smoke test against https://\${APP_DOMAIN}

Sized for Ubuntu 24.04, t3.medium (4 GB RAM), disk ≥ 30 GB.
The ai-service image includes torch + sentence-transformers + spaCy es_core_news_md;
measure with \`docker image ls\` before tightening disk.

EOF
