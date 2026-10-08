param(
    [string]$Target = "orion.local",
    [string]$RemoteRoot = "/opt/orion",
    [string]$Service = "orion.service",
    [string]$RemotePython = "/opt/orion/.venv-voice/bin/python",
    [string]$RemoteUv = "/usr/local/bin/uv",
    [string]$RemoteLlamaServer = "/usr/local/bin/llama-server",
    [string]$SshIdentityFile,
    [string]$DatabasePath = "/var/lib/orion/orion.sqlite3",
    [switch]$EnableJarvisWakeWord,
    [ValidateRange(0.0, 1.0)]
    [double]$WakeWordThreshold = 0.08
)

$ErrorActionPreference = "Stop"

function Assert-ExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Quote-Remote([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Invoke-Pi([string]$Command, [string]$Step, [switch]$NoSudo) {
    if ($NoSudo) {
        & ssh.exe @script:SshOptions $Target $Command
        Assert-ExitCode $Step
        return
    }

    if ($null -eq $script:PiSudoPassword) {
        $script:PiSudoPassword = Read-Host "Enter Pi sudo password (kept in memory for this run only)" -AsSecureString
    }

    $passwordPointer = [IntPtr]::Zero
    $passwordText = $null
    try {
        $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($script:PiSudoPassword)
        $passwordText = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
        $encodedCommand = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Command))
        $remoteCommand = 'IFS= read -r ORION_SUDO_PASSWORD || exit 1; ORION_SUDO_PASSWORD=$(printf ''%s'' "$ORION_SUDO_PASSWORD" | tr -d ''\r''); printf ''%s\n'' "$ORION_SUDO_PASSWORD" | command sudo -S -p ORION_SUDO_PROMPT -v || exit $?; unset ORION_SUDO_PASSWORD; ORION_DEPLOY_COMMAND=$(printf ''%s'' ' + $encodedCommand + ' | base64 -d) || exit 1; eval "$ORION_DEPLOY_COMMAND"'
        $sshCommand = $remoteCommand
        if ($PSVersionTable.PSVersion -lt [version]"7.3" -or $PSNativeCommandArgumentPassing -eq "Legacy") {
            $sshCommand = $sshCommand.Replace('"', '\"')
        }
        $passwordText | & ssh.exe @script:SshOptions -T $Target $sshCommand
    }
    finally {
        $passwordText = $null
        if ($passwordPointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
        }
    }
    Assert-ExitCode $Step
}

function Add-SourceDirectory(
    [string]$Root,
    [string]$Directory,
    [System.IO.Compression.ZipArchive]$Archive,
    [string[]]$ExcludedDirectories,
    [string[]]$ExcludedFiles
) {
    foreach ($item in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($item.PSIsContainer) {
            if ($ExcludedDirectories -contains $item.Name) { continue }
            [void](Add-SourceDirectory $Root $item.FullName $Archive $ExcludedDirectories $ExcludedFiles)
            continue
        }

        $fileName = $item.Name
        if ($fileName -ne "orion.env.example" -and $fileName -match '(^|\.)env($|\.)') { continue }
        if ($fileName -like "*secret*" -or $fileName -like "*.key" -or $fileName -like "*.pem") { continue }
        if (@($ExcludedFiles | Where-Object { $fileName -like $_ }).Count -gt 0) { continue }

        $rootPrefix = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
        $relative = $item.FullName.Substring($rootPrefix.Length)
        $entryName = $relative.Replace('\', '/')
        $entry = $Archive.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Fastest)
        $inputStream = [System.IO.File]::OpenRead($item.FullName)
        $outputStream = $entry.Open()
        try { $inputStream.CopyTo($outputStream) }
        finally {
            $outputStream.Dispose()
            $inputStream.Dispose()
        }
    }
}

foreach ($command in @("ssh.exe", "scp.exe")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required. Install OpenSSH Client or add it to PATH."
    }
}

if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteRoot -eq "/") {
    throw "RemoteRoot must be an absolute, plain path such as /opt/orion."
}
if ($Service -notmatch '^[A-Za-z0-9_.@-]+\.service$') {
    throw "Service must be a systemd unit name ending in .service."
}
if ($DatabasePath -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "DatabasePath must be an absolute path without shell-special characters."
}
if ($RemoteLlamaServer -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "RemoteLlamaServer must be an absolute path without shell-special characters."
}
$root = Split-Path -Parent $PSScriptRoot
$runId = [guid]::NewGuid().ToString("N")
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) "orion-$runId.zip"
$remoteStage = "/tmp/orion-deploy-$runId"
$remoteArchive = "$remoteStage.zip"
$remoteBackup = "/var/backups/orion/deploy-$runId"
$remoteConfigBackup = "/var/backups/orion/deploy-$runId-config"
$remoteEnvironmentBackup = "$remoteConfigBackup/orion.env"
$remoteUnitBackup = "$remoteConfigBackup/$Service"
$remotePythonInstall = "$RemoteRoot/python"
$remoteEnvironment = "$RemoteRoot/.venv-voice"
$remoteModel = "$RemoteRoot/models/Qwen3.5-2B-Q4_K_M.gguf"
$modelSha256 = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
$llamaCommit = "d81235049384534c167caea52b85a694f6103d14"
$serviceQuoted = Quote-Remote $Service
$rootQuoted = Quote-Remote $RemoteRoot
$pythonQuoted = Quote-Remote $RemotePython
$uvQuoted = Quote-Remote $RemoteUv
$stageQuoted = Quote-Remote $remoteStage
$backupQuoted = Quote-Remote $remoteBackup
$configBackupQuoted = Quote-Remote $remoteConfigBackup
$environmentBackupQuoted = Quote-Remote $remoteEnvironmentBackup
$unitBackupQuoted = Quote-Remote $remoteUnitBackup
$database = "/tmp/orion-deploy-$runId.sqlite3"
$testHome = "/tmp/orion-deploy-home-$runId"
$databaseQuoted = Quote-Remote $database
$databasePathQuoted = Quote-Remote $DatabasePath
$temporaryDatabaseBackup = "/var/lib/orion/.orion-deploy-$runId.sqlite3"
$temporaryDatabaseBackupQuoted = Quote-Remote $temporaryDatabaseBackup
$testHomeQuoted = Quote-Remote $testHome
$wakeWordThresholdValue = $WakeWordThreshold.ToString(
    "0.###",
    [System.Globalization.CultureInfo]::InvariantCulture
)
$backupCreated = $false
$deploymentChanged = $false
$serviceStopped = $false
$zip = $null
$archiveStream = $null
$script:PiSudoPassword = $null
$script:SshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
if (-not [string]::IsNullOrWhiteSpace($SshIdentityFile)) {
    try {
        $resolvedIdentity = (Resolve-Path -LiteralPath $SshIdentityFile -ErrorAction Stop).Path
    }
    catch {
        throw "SSH identity file not found: $SshIdentityFile"
    }
    if (-not (Test-Path -LiteralPath $resolvedIdentity -PathType Leaf)) {
        throw "SSH identity file must be a file: $resolvedIdentity"
    }
    $script:SshOptions += @("-i", $resolvedIdentity, "-o", "IdentitiesOnly=yes")
}
$remoteConnectionVerified = $false
$llamaCommitShort = $llamaCommit.Substring(0, 7)
$modelEnvironmentPython = @'
import os, pathlib, re, sys, tempfile
path = pathlib.Path(sys.argv[1])
values = {
    "ORION_MODEL_BACKEND": "llama.cpp",
    "ORION_MODEL_BASE_URL": "http://127.0.0.1:8080/v1",
    "ORION_MODEL_PATH": sys.argv[2],
    "ORION_LLAMA_SERVER_PATH": sys.argv[3],
    "ORION_MODEL_CONTEXT_SIZE": "4096",
    "ORION_MAX_CONTEXT_TOKENS": "3072",
    "ORION_MODEL_ANSWER_TOKENS": "512",
    "ORION_MODEL_PLANNER_TOKENS": "128",
}
original = path.stat()
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
updated = []
for line in lines:
    match = re.match(r"\s*#?\s*([A-Z0-9_]+)\s*=", line)
    key = match.group(1) if match else None
    if key in values:
        if key not in seen:
            updated.append(f"{key}={values[key]}")
            seen.add(key)
    else:
        updated.append(line)
updated.extend(f"{key}={value}" for key, value in values.items() if key not in seen)
fd, temporary = tempfile.mkstemp(prefix=".orion.env.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(updated) + "\n")
    os.chmod(temporary, original.st_mode)
    os.chown(temporary, original.st_uid, original.st_gid)
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
'@
$modelEnvironmentBase64 = [Convert]::ToBase64String(
    [System.Text.Encoding]::UTF8.GetBytes($modelEnvironmentPython)
)

try {
    & ssh.exe @script:SshOptions $Target "true"
    Assert-ExitCode "SSH key authentication preflight"
    $remoteConnectionVerified = $true

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archiveStream = [System.IO.File]::Open(
        $archivePath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    $zip = [System.IO.Compression.ZipArchive]::new(
        $archiveStream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $true
    )

    $excludedDirectories = @(
        ".git", ".venv-local", ".venv", ".venv-voice", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "models", "python",
        "dist", "build", "orion_assistant.egg-info"
    )
    $excludedFiles = @("*.pyc", "*.pyo", "*.sqlite", "*.sqlite-*", "*.sqlite3", "*.sqlite3-*", "*.db", "*.db-*", "*.sqlite3-wal", "*.sqlite3-shm")

    [void](Add-SourceDirectory $root $root $zip $excludedDirectories $excludedFiles)
    if ($EnableJarvisWakeWord) {
        $jarvisModel = Join-Path $root "models/hey_jarvis.onnx"
        if (-not (Test-Path -LiteralPath $jarvisModel -PathType Leaf)) {
            throw "Jarvis model not found: $jarvisModel"
        }
        $modelEntry = $zip.CreateEntry("models/hey_jarvis.onnx", [System.IO.Compression.CompressionLevel]::Fastest)
        $modelInput = [System.IO.File]::OpenRead($jarvisModel)
        $modelOutput = $modelEntry.Open()
        try { $modelInput.CopyTo($modelOutput) }
        finally {
            $modelOutput.Dispose()
            $modelInput.Dispose()
        }
    }
    $zip.Dispose()
    $zip = $null
    $archiveStream.Dispose()
    $archiveStream = $null

    Write-Host "Uploading source archive to $Target..."
    & ssh.exe @script:SshOptions $Target "mkdir -p $(Quote-Remote $remoteStage)"
    Assert-ExitCode "Remote staging directory creation"
    & scp.exe @script:SshOptions $archivePath "${Target}:$remoteArchive"
    Assert-ExitCode "Source upload"
    Invoke-Pi "unzip -q $(Quote-Remote $remoteArchive) -d $(Quote-Remote $remoteStage)" "Source extraction" -NoSudo

    Invoke-Pi "if test -x $(Quote-Remote $RemoteLlamaServer); then :; else echo 'Missing executable llama-server: $RemoteLlamaServer' >&2; exit 1; fi" "llama-server executable check" -NoSudo
    Invoke-Pi "if test -f $(Quote-Remote $remoteModel); then :; else echo 'Missing Qwen GGUF: $remoteModel' >&2; exit 1; fi" "Qwen GGUF file check" -NoSudo
    Invoke-Pi "printf '%s  %s\n' $(Quote-Remote $modelSha256) $(Quote-Remote $remoteModel) | sha256sum --check" "Qwen GGUF SHA-256 check" -NoSudo
    Invoke-Pi "if $(Quote-Remote $RemoteLlamaServer) --version 2>&1 | grep -F $(Quote-Remote $llamaCommitShort) >/dev/null; then :; else echo 'llama-server version does not match pinned commit $llamaCommitShort' >&2; exit 1; fi" "llama.cpp build revision check" -NoSudo

    Write-Host "Stopping $Service..."
    Invoke-Pi "sudo systemctl stop $serviceQuoted" "Service stop"
    $serviceStopped = $true

    $preflight = @(
        "cd $stageQuoted",
        "sudo install -d -o orion-service -g orion-service -m 0700 $testHomeQuoted",
        "sudo -u orion-service env HOME=$testHomeQuoted ORION_DATABASE_PATH=$databaseQuoted PYTHONPATH=$(Quote-Remote "$remoteStage/src") $pythonQuoted -m pytest -p no:cacheprovider -q $(Quote-Remote "$remoteStage/tests")",
        "sudo -u orion-service env HOME=$testHomeQuoted ORION_DATABASE_PATH=$databaseQuoted PYTHONPATH=$(Quote-Remote "$remoteStage/src") $pythonQuoted -m orion --check-config"
    ) -join " && "
    Invoke-Pi $preflight "Staged tests and configuration check"

    $backupCommand = @(
        "sudo install -d -m 0700 $(Quote-Remote "/var/backups/orion")",
        "sudo install -d -m 0700 $backupQuoted",
        "sudo install -d -m 0700 $configBackupQuoted",
        "sudo cp -a $(Quote-Remote "/etc/orion/orion.env") $environmentBackupQuoted",
        "sudo cp -a $(Quote-Remote "/etc/systemd/system/$Service") $unitBackupQuoted",
        "sudo rsync -a --delete --chown=root:orion-service --exclude=.venv-local/ --exclude=.venv/ --exclude=.venv-voice/ --exclude=models/ --exclude=llama.cpp/ --exclude=python/ --exclude=.git/ --exclude=src/orion_assistant.egg-info/ --exclude='__pycache__/' --exclude='*.sqlite3' --exclude='*.sqlite3-wal' --exclude='*.sqlite3-shm' $rootQuoted/ $backupQuoted/"
    ) -join " && "
    Invoke-Pi $backupCommand "Source backup"
    if ($EnableJarvisWakeWord) {
        $jarvisBackup = @(
            "if sudo test -f $(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx"); then sudo install -D -m 0640 $(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx") $backupQuoted/models/hey_jarvis.onnx; fi"
        ) -join " && "
        Invoke-Pi $jarvisBackup "Wake-word environment and model backup"
    }
    $backupCreated = $true

    $databaseBackupPython = 'import sqlite3,sys; source=sqlite3.connect(sys.argv[1]); target=sqlite3.connect(sys.argv[2]); source.backup(target); target.close(); source.close()'
    $databaseBackupCommand = "if sudo test -f $databasePathQuoted; then sudo -u orion-service env HOME=/var/lib/orion $pythonQuoted -c $(Quote-Remote $databaseBackupPython) $databasePathQuoted $temporaryDatabaseBackupQuoted && sudo install -o root -g root -m 0600 $temporaryDatabaseBackupQuoted $backupQuoted/orion.sqlite3 && sudo rm -f $temporaryDatabaseBackupQuoted; fi"
    Invoke-Pi $databaseBackupCommand "SQLite online backup"

    $deployCommand = @(
        "sudo rsync -a --delete --chown=root:orion-service --exclude=.venv-local/ --exclude=.venv/ --exclude=.venv-voice/ --exclude=models/ --exclude=llama.cpp/ --exclude=python/ --exclude=.git/ --exclude=src/orion_assistant.egg-info/ --exclude='__pycache__/' --exclude='*.sqlite3' --exclude='*.sqlite3-wal' --exclude='*.sqlite3-shm' $stageQuoted/ $rootQuoted/",
        "sudo chmod 0750 $(Quote-Remote "$RemoteRoot/scripts/phase11-capture.sh")",
        "sudo install -d -o orion-service -g orion-service $(Quote-Remote "$RemoteRoot/src/orion_assistant.egg-info")",
        "sudo chown -R orion-service:orion-service $(Quote-Remote "$RemoteRoot/src/orion_assistant.egg-info")",
        "sudo install -d -o orion-service -g orion-service $(Quote-Remote "$RemoteRoot/build")",
        "sudo chown -R orion-service:orion-service $(Quote-Remote "$RemoteRoot/build")",
        "sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=$(Quote-Remote $remotePythonInstall) UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=$(Quote-Remote $remoteEnvironment) $uvQuoted sync --locked --no-editable --no-install-package tflite-runtime --extra voice --extra wakeword --extra pi --extra test --python 3.12 --project $rootQuoted",
        "cd $rootQuoted"
    )
    if ($EnableJarvisWakeWord) {
        $deployCommand += @(
            "sudo install -d -o orion-service -g orion-service -m 0750 $(Quote-Remote "$RemoteRoot/models")",
            "sudo install -o orion-service -g orion-service -m 0640 $(Quote-Remote "$remoteStage/models/hey_jarvis.onnx") $(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx")"
        )
    }
    $deployCommand += @(
        "sudo install -o root -g root -m 0644 $(Quote-Remote "$RemoteRoot/systemd/orion.service") $(Quote-Remote "/etc/systemd/system/$Service")",
        "printf %s $(Quote-Remote $modelEnvironmentBase64) | base64 -d | sudo python3 - $(Quote-Remote "/etc/orion/orion.env") $(Quote-Remote $remoteModel) $(Quote-Remote $RemoteLlamaServer)",
        "sudo -u orion-service env HOME=$testHomeQuoted ORION_DATABASE_PATH=$databaseQuoted PYTHONPATH=$(Quote-Remote "$RemoteRoot/src") $pythonQuoted -m pytest -p no:cacheprovider -q $(Quote-Remote "$RemoteRoot/tests")",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable $serviceQuoted"
    )
    if ($EnableJarvisWakeWord) {
        $jarvisEnvironmentPython = @'
import os, pathlib, re, sys, tempfile
path = pathlib.Path(sys.argv[1])
values = {
    "ORION_VOICE_PROFILE": "legacy",
    "ORION_STT_BACKEND": "vosk",
    "ORION_AUDIO_DEVICE": "sysdefault",
    "ORION_WAKE_WORD_ENABLED": "true",
    "ORION_WAKE_WORD_MODEL_PATH": "/opt/orion/models/hey_jarvis.onnx",
    "ORION_WAKE_WORD_MELSPECTROGRAM_PATH": "/opt/orion/artifacts/wakeword-staging/melspectrogram.onnx",
    "ORION_WAKE_WORD_EMBEDDING_PATH": "/opt/orion/artifacts/wakeword-staging/embedding_model.onnx",
    "ORION_WAKE_WORD_THRESHOLD": "__WAKE_WORD_THRESHOLD__",
    "ORION_WAKE_WORD_COOLDOWN_SECONDS": "3.0",
}
original = path.stat()
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
updated = []
for line in lines:
    match = re.match(r"\s*#?\s*([A-Z0-9_]+)\s*=", line)
    key = match.group(1) if match else None
    if key in values:
        if key not in seen:
            updated.append(f"{key}={values[key]}")
            seen.add(key)
    else:
        updated.append(line)
updated.extend(f"{key}={value}" for key, value in values.items() if key not in seen)
fd, temporary = tempfile.mkstemp(prefix=".orion.env.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(updated) + "\n")
    os.chmod(temporary, original.st_mode)
    os.chown(temporary, original.st_uid, original.st_gid)
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
'@
        $jarvisEnvironmentPython = $jarvisEnvironmentPython.Replace(
            "__WAKE_WORD_THRESHOLD__",
            $wakeWordThresholdValue
        )
        $jarvisEnvironmentBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($jarvisEnvironmentPython))
        $deployCommand += @(
            "printf %s $(Quote-Remote $jarvisEnvironmentBase64) | base64 -d | sudo python3 - $(Quote-Remote "/etc/orion/orion.env")",
            "sudo -u orion-service env HOME=/var/lib/orion ORION_VOICE_PROFILE=legacy ORION_STT_BACKEND=vosk ORION_AUDIO_DEVICE=sysdefault ORION_WAKE_WORD_ENABLED=true ORION_WAKE_WORD_MODEL_PATH=$(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx") ORION_WAKE_WORD_MELSPECTROGRAM_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/melspectrogram.onnx") ORION_WAKE_WORD_EMBEDDING_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/embedding_model.onnx") ORION_WAKE_WORD_THRESHOLD=$wakeWordThresholdValue ORION_WAKE_WORD_COOLDOWN_SECONDS=3.0 $pythonQuoted -m orion --validate-wakeword-assets",
            "sudo -u orion-service env HOME=/var/lib/orion ORION_VOICE_PROFILE=legacy ORION_STT_BACKEND=vosk ORION_AUDIO_DEVICE=sysdefault ORION_WAKE_WORD_ENABLED=true ORION_WAKE_WORD_MODEL_PATH=$(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx") ORION_WAKE_WORD_MELSPECTROGRAM_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/melspectrogram.onnx") ORION_WAKE_WORD_EMBEDDING_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/embedding_model.onnx") ORION_WAKE_WORD_THRESHOLD=$wakeWordThresholdValue ORION_WAKE_WORD_COOLDOWN_SECONDS=3.0 $pythonQuoted -c $(Quote-Remote "import numpy as np; from orion.audio.wakeword import OpenWakeWordDetector; from orion.config import Settings; d=OpenWakeWordDetector.from_settings(Settings.from_environment()); a=np.zeros(16000 * 3, dtype=np.int16); s=[d._model.predict(a[i:i+1280]) for i in range(0, len(a), 1280)]; assert len(s) == 38 and all(isinstance(x, dict) and x for x in s); print('Jarvis ONNX inference passed: 3 seconds, 38 frames')")"
        )
    }
    $deployCommand += @(
        "sudo systemctl start $serviceQuoted",
        "sudo systemctl is-active --quiet $serviceQuoted"
    )
    Invoke-Pi ($deployCommand -join " && ") "Deployment, locked environment sync, tests, and service start"
    $deploymentChanged = $true
    $serviceStopped = $false

    Write-Host "Deployment succeeded. Recent service log:"
    try {
        Invoke-Pi "sudo journalctl -u $serviceQuoted -n 25 --no-pager" "Recent service log"
    }
    catch {
        Write-Warning "Could not read recent service logs. The service is active; inspect it with journalctl."
    }
    Write-Host "Source backup retained at $remoteBackup; systemd and environment backup retained at $remoteConfigBackup"
    Write-Warning "AIRHUG 01 PortAudio naming and sample rates still require Pi verification."
}
catch {
    $failure = $_
    if ($backupCreated) {
        Write-Warning "Deployment failed. Restoring source backup and previous service."
        $rollback = @(
            "sudo systemctl stop $serviceQuoted",
            "sudo cp -a $environmentBackupQuoted $(Quote-Remote "/etc/orion/orion.env")",
            "sudo cp -a $unitBackupQuoted $(Quote-Remote "/etc/systemd/system/$Service")",
            "sudo rsync -a --delete --chown=root:orion-service --exclude=.venv-local/ --exclude=.venv/ --exclude=.venv-voice/ --exclude=models/ --exclude=llama.cpp/ --exclude=python/ --exclude=.git/ --exclude=src/orion_assistant.egg-info/ --exclude='__pycache__/' --exclude='*.sqlite3' --exclude='*.sqlite3-wal' --exclude='*.sqlite3-shm' $backupQuoted/ $rootQuoted/",
            "sudo chmod 0750 $(Quote-Remote "$RemoteRoot/scripts/phase11-capture.sh")",
            "sudo install -d -o orion-service -g orion-service $(Quote-Remote "$RemoteRoot/src/orion_assistant.egg-info")",
            "sudo chown -R orion-service:orion-service $(Quote-Remote "$RemoteRoot/src/orion_assistant.egg-info")",
            "sudo install -d -o orion-service -g orion-service $(Quote-Remote "$RemoteRoot/build")",
            "sudo chown -R orion-service:orion-service $(Quote-Remote "$RemoteRoot/build")",
            "sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=$(Quote-Remote $remotePythonInstall) UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=$(Quote-Remote $remoteEnvironment) $uvQuoted sync --locked --no-editable --no-install-package tflite-runtime --extra voice --extra wakeword --extra pi --extra test --python 3.12 --project $rootQuoted",
            "sudo systemctl daemon-reload"
        )
        if ($EnableJarvisWakeWord) {
            $rollback += @(
                "if sudo test -f $backupQuoted/models/hey_jarvis.onnx; then sudo install -d -o orion-service -g orion-service -m 0750 $(Quote-Remote "$RemoteRoot/models") && sudo install -o orion-service -g orion-service -m 0640 $backupQuoted/models/hey_jarvis.onnx $(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx"); else sudo rm -f $(Quote-Remote "$RemoteRoot/models/hey_jarvis.onnx"); fi"
            )
        }
        $rollback += "sudo systemctl start $serviceQuoted"
        try {
            Invoke-Pi ($rollback -join " && ") "Automatic rollback"
        }
        catch {
            Write-Warning "Automatic rollback command failed. Check service state; source backup is $remoteBackup."
        }
    }
    elseif ($serviceStopped -and -not $deploymentChanged -and $remoteConnectionVerified) {
        Write-Warning "Deployment was not changed. Attempting to restart the original service."
        try {
            Invoke-Pi "sudo systemctl start $serviceQuoted" "Restart original service"
            $serviceStopped = $false
        }
        catch {
            Write-Warning "Could not restart the original service. Inspect it on the Pi."
        }
    }
    throw $failure
}
finally {
    if ($null -ne $zip) { $zip.Dispose() }
    if ($null -ne $archiveStream) { $archiveStream.Dispose() }
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    if ($null -ne $script:PiSudoPassword) {
        try {
            Invoke-Pi "sudo rm -f $(Quote-Remote $temporaryDatabaseBackup)" "Temporary database backup cleanup"
        }
        catch {
            Write-Warning "Could not remove temporary database backup $temporaryDatabaseBackup."
        }
    }
    try {
        $cleanupCommand = "rm -rf $(Quote-Remote $remoteStage) $(Quote-Remote $remoteArchive) $(Quote-Remote $testHome) $(Quote-Remote $database) $(Quote-Remote "$database-wal") $(Quote-Remote "$database-shm")"
        if ($null -ne $script:PiSudoPassword) {
            Invoke-Pi "sudo $cleanupCommand" "Remote staging cleanup"
        }
        else {
            Invoke-Pi $cleanupCommand "Remote staging cleanup" -NoSudo
        }
    }
    catch {
        Write-Warning "Could not remove remote staging files. Remove /tmp/orion-deploy-$runId* on the Pi after recovery."
    }
    finally {
        if ($null -ne $script:PiSudoPassword) {
            $script:PiSudoPassword.Dispose()
            $script:PiSudoPassword = $null
        }
    }
}