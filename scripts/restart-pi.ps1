param(
    [string]$Target = "orion.local",
    [string]$Service = "orion.service"
)

$ErrorActionPreference = "Stop"
if ($Service -notmatch '^[A-Za-z0-9_.@-]+\.service$') {
    throw "Service must be a systemd unit name ending in .service."
}
if (-not (Get-Command "ssh.exe" -ErrorAction SilentlyContinue)) {
    throw "ssh.exe is required. Install OpenSSH Client or add it to PATH."
}

& ssh.exe -tt -o ConnectTimeout=10 $Target "sudo systemctl restart '$Service' && sudo systemctl is-active '$Service' && sudo journalctl -u '$Service' -n 25 --no-pager"
if ($LASTEXITCODE -ne 0) {
    throw "Restart or verification failed with exit code $LASTEXITCODE."
}