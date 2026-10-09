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
    [switch]$DisableJarvisWakeWord,
    [ValidateRange(0.0, 1.0)]
    [double]$WakeWordThreshold = 0.08,
    [string]$RemoteModelRoot,
    [string]$LocalModelRoot
)

$ErrorActionPreference = "Stop"
$UseJarvisWakeWord = $EnableJarvisWakeWord -or -not $DisableJarvisWakeWord

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
        $remoteCommand = 'IFS= read -r ORION_SUDO_PASSWORD || exit 1; ORION_SUDO_PASSWORD=$(printf ''%s'' "$ORION_SUDO_PASSWORD" | tr -d ''\r''); printf ''%s\n'' "$ORION_SUDO_PASSWORD" | command sudo -S -p "" -v || exit $?; unset ORION_SUDO_PASSWORD; sudo() { command sudo -n "$@"; }; ORION_DEPLOY_COMMAND=$(printf ''%s'' ' + $encodedCommand + ' | base64 -d) || exit 1; eval "$ORION_DEPLOY_COMMAND"'
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
if ([string]::IsNullOrWhiteSpace($RemoteModelRoot)) {
    $RemoteModelRoot = "$RemoteRoot/models"
}
if ($RemoteModelRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteModelRoot -eq "/") {
    throw "RemoteModelRoot must be an absolute, plain path such as /opt/orion/models."
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
if ([string]::IsNullOrWhiteSpace($LocalModelRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:ORION_MODEL_ROOT)) {
        $LocalModelRoot = $env:ORION_MODEL_ROOT
    }
    else {
        $LocalModelRoot = Join-Path (Split-Path -Parent $root) "ORION-MODELS"
    }
}
if (-not (Test-Path -LiteralPath $LocalModelRoot -PathType Container)) {
    throw "Local model depot not found: $LocalModelRoot. Set -LocalModelRoot or ORION_MODEL_ROOT."
}
$LocalModelRoot = (Resolve-Path -LiteralPath $LocalModelRoot).Path
$runId = [guid]::NewGuid().ToString("N")
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) "orion-$runId.zip"
$remoteStage = "/tmp/orion-deploy-$runId"
$remoteArchive = "$remoteStage.zip"
$remoteModelStage = "/home/orion/.cache/orion-deploy-$runId/model-assets"
$remoteModelStageParent = "/home/orion/.cache/orion-deploy-$runId"
$localModelStageParent = Join-Path (Split-Path -Parent $LocalModelRoot) ".orion-model-upload-$runId"
$localModelStage = Join-Path $localModelStageParent "model-assets"
$remoteBackup = "/var/backups/orion/deploy-$runId"
$remoteConfigBackup = "/var/backups/orion/deploy-$runId-config"
$remoteEnvironmentBackup = "$remoteConfigBackup/orion.env"
$remoteUnitBackup = "$remoteConfigBackup/$Service"
$remotePythonInstall = "$RemoteRoot/python"
$remoteEnvironment = "$RemoteRoot/.venv-voice"
$remoteModel = "$RemoteModelRoot/LLM/Qwen3.5-2B-Q4_K_M.gguf"
$legacyRemoteModel = "$RemoteModelRoot/Qwen3.5-2B-Q4_K_M.gguf"
$modelSha256 = "a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b"
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
$script:SshOptions = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3"
)
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
    "ORION_MODEL_ROOT": sys.argv[5],
    "ORION_MODEL_PATH": next((candidate for candidate in sys.argv[2:4] if pathlib.Path(candidate).is_file()), None),
    "ORION_LLAMA_SERVER_PATH": sys.argv[4],
    "ORION_MODEL_CONTEXT_SIZE": "4096",
    "ORION_MAX_CONTEXT_TOKENS": "3072",
    "ORION_MODEL_ANSWER_TOKENS": "512",
    "ORION_MODEL_PLANNER_TOKENS": "128",
    "ORION_VOICE_PROFILE": "legacy",
    "ORION_STT_BACKEND": "vosk",
    "ORION_STT_MODEL_ID": "vosk-en-small",
    "ORION_VOSK_MODEL_PATH": str(pathlib.Path(sys.argv[5]) / "vosk-model-small-en-us-0.15"),
    "ORION_TTS_BACKEND": "piper",
    "ORION_TTS_MODEL_ID": "piper-lessac-medium",
    "ORION_WAKEWORD_MODEL_ID": "hey-jarvis",
    "ORION_WAKE_WORD_ENABLED": "false",
}
if values["ORION_MODEL_PATH"] is None:
    raise SystemExit("Q4 GGUF is missing from canonical and legacy model paths")
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
    if ($UseJarvisWakeWord) {
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
    $modelCheck = "if test -f $(Quote-Remote $remoteModel); then printf '%s  %s\n' $(Quote-Remote $modelSha256) $(Quote-Remote $remoteModel) | sha256sum --check; elif test -f $(Quote-Remote $legacyRemoteModel); then printf '%s  %s\n' $(Quote-Remote $modelSha256) $(Quote-Remote $legacyRemoteModel) | sha256sum --check; else echo 'Missing Qwen GGUF under model root $RemoteModelRoot' >&2; exit 1; fi"
    Invoke-Pi $modelCheck "Qwen GGUF file and SHA-256 check" -NoSudo
    Invoke-Pi "if $(Quote-Remote $RemoteLlamaServer) --version 2>&1 | grep -F $(Quote-Remote $llamaCommitShort) >/dev/null; then :; else echo 'llama-server version does not match pinned commit $llamaCommitShort' >&2; exit 1; fi" "llama.cpp build revision check" -NoSudo

    $modelAssetSources = @(
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "LLM/Qwen3.5-2B-Q4_K_M.gguf"; RelativePath = "LLM/Qwen3.5-2B-Q4_K_M.gguf" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "STT/ggml-base.en.bin"; RelativePath = "STT/ggml-base.en.bin" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "TTS/en_US-lessac-medium.onnx"; RelativePath = "TTS/en_US-lessac-medium.onnx" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "TTS/en_US-lessac-medium.onnx.json"; RelativePath = "TTS/en_US-lessac-medium.onnx.json" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "EMBEDDING/BGE-small-en-v1.5-model.onnx"; RelativePath = "EMBEDDING/BGE-small-en-v1.5-model.onnx" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "WAKEWORD/hey_orion.onnx"; RelativePath = "WAKEWORD/hey_orion.onnx" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "WAKEWORD/hey_orion.onnx.data"; RelativePath = "WAKEWORD/hey_orion.onnx.data" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "WAKEWORD/hey_orion.yaml"; RelativePath = "WAKEWORD/hey_orion.yaml" },
        [pscustomobject]@{ SourcePath = Join-Path $LocalModelRoot "WAKEWORD/SHA256.txt"; RelativePath = "WAKEWORD/SHA256.txt" },
        [pscustomobject]@{ SourcePath = Join-Path $root "artifacts/wakeword-staging/melspectrogram.onnx"; RelativePath = "WAKEWORD/melspectrogram.onnx" },
        [pscustomobject]@{ SourcePath = Join-Path $root "artifacts/wakeword-staging/embedding_model.onnx"; RelativePath = "WAKEWORD/embedding_model.onnx" }
    )
    $localWakeManifest = Join-Path $LocalModelRoot "WAKEWORD/phase13-artifacts.json"
    if (-not (Test-Path -LiteralPath $localWakeManifest -PathType Leaf)) {
        $localWakeManifest = Join-Path $root "artifacts/wakeword-staging/hey_orion_v0_6_0/phase13-artifacts.json"
    }
    if (Test-Path -LiteralPath $localWakeManifest -PathType Leaf) {
        $modelAssetSources += [pscustomobject]@{ SourcePath = $localWakeManifest; RelativePath = "WAKEWORD/phase13-artifacts.json" }
    }
    else {
        Write-Warning "Hey Orion manifest is absent. Model files will upload, but wakeword mode remains blocked by asset validation."
    }
    $selectedModelFiles = @(
        foreach ($asset in $modelAssetSources) {
            if (-not (Test-Path -LiteralPath $asset.SourcePath -PathType Leaf)) {
                throw "Required model asset missing: $($asset.SourcePath)"
            }
            [pscustomobject]@{
                FullName = (Resolve-Path -LiteralPath $asset.SourcePath).Path
                RelativePath = $asset.RelativePath
                Length = (Get-Item -LiteralPath $asset.SourcePath).Length
            }
        }
    )
    $wakeHashFile = Join-Path $LocalModelRoot "WAKEWORD/SHA256.txt"
    $wakeHashLine = (Get-Content -LiteralPath $wakeHashFile -Raw).Trim()
    if ($wakeHashLine -notmatch '^([0-9a-fA-F]{64})\s+hey_orion\.onnx$') {
        throw "Invalid Hey Orion checksum record: $wakeHashFile"
    }
    $wakeModelPath = Join-Path $LocalModelRoot "WAKEWORD/hey_orion.onnx"
    $wakeActualHash = (Get-FileHash -LiteralPath $wakeModelPath -Algorithm SHA256).Hash
    if ($wakeActualHash -ine $Matches[1]) {
        throw "Hey Orion SHA-256 mismatch: $wakeModelPath"
    }
    $remoteModelFiles = @(Invoke-Pi "sudo find $(Quote-Remote $RemoteModelRoot) -type f -printf '%P\t%s\n'" "Pi model asset inventory")
    $remoteModelSizes = [System.Collections.Generic.Dictionary[string, long]]::new([System.StringComparer]::Ordinal)
    foreach ($inventoryLine in $remoteModelFiles) {
        $parts = ([string]$inventoryLine) -split "`t", 2
        if ($parts.Count -eq 2 -and $parts[1] -match '^\d+$') {
            $remoteModelSizes[$parts[0]] = [long]$parts[1]
        }
    }
    $missingModelFiles = @(
        $selectedModelFiles | Where-Object {
            -not $remoteModelSizes.ContainsKey($_.RelativePath) -or $remoteModelSizes[$_.RelativePath] -ne $_.Length
        }
    )
    if ($missingModelFiles.Count -gt 0) {
        Write-Host "Transferring $($missingModelFiles.Count) model-depot files missing from Pi storage or with a size mismatch:"
        $modelStageRoot = $remoteModelStage
        New-Item -ItemType Directory -Path $localModelStage -Force | Out-Null
        foreach ($file in $missingModelFiles) {
            $relativePath = $file.RelativePath
            Write-Host "  $relativePath"
            $localRelativePath = $relativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
            $localStagedFile = Join-Path $localModelStage $localRelativePath
            New-Item -ItemType Directory -Path (Split-Path -Parent $localStagedFile) -Force | Out-Null
            New-Item -ItemType HardLink -Path $localStagedFile -Target $file.FullName | Out-Null
        }
        Invoke-Pi "mkdir -p $(Quote-Remote $remoteModelStageParent)" "Model asset staging directory creation" -NoSudo
        & scp.exe @script:SshOptions -X nrequests=128 -X buffer=65536 -r $localModelStage "${Target}:$remoteModelStageParent/"
        Assert-ExitCode "Missing model asset upload"
        $installModelCommands = @(
            "sudo install -d -o root -g orion-service -m 0755 $(Quote-Remote $RemoteModelRoot)"
        )
        foreach ($file in $missingModelFiles) {
            $relativePath = $file.RelativePath
            $relativeDirectory = Split-Path -Parent $relativePath
            $remoteDestination = if ($relativeDirectory) { "$RemoteModelRoot/$relativeDirectory" } else { $RemoteModelRoot }
            $remoteStagedFile = "$modelStageRoot/$relativePath"
            $remoteDestinationFile = "$remoteDestination/$(Split-Path -Leaf $relativePath)"
            $expectedSize = $file.Length
            $installModelCommands += "sudo install -d -o root -g orion-service -m 0755 $(Quote-Remote $remoteDestination) && if sudo test -f $(Quote-Remote $remoteDestinationFile) && test `"`$(sudo stat -c %s $(Quote-Remote $remoteDestinationFile))`" = '$expectedSize'; then rm -f $(Quote-Remote $remoteStagedFile); else sudo rsync -a --chown=root:orion-service --chmod=D755,F644 $(Quote-Remote $remoteStagedFile) $(Quote-Remote $remoteDestinationFile) && rm -f $(Quote-Remote $remoteStagedFile); fi"
        }
        Invoke-Pi ($installModelCommands -join " && ") "Missing model asset installation"
    }
    else {
        Write-Host "All local model-depot files are already present on Pi."
    }

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
    if ($UseJarvisWakeWord) {
        $jarvisBackup = @(
            "if sudo test -f $(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx"); then sudo install -D -m 0640 $(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx") $backupQuoted/models/hey_jarvis.onnx; fi"
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
        "sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=$(Quote-Remote $remotePythonInstall) UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=$(Quote-Remote $remoteEnvironment) $uvQuoted sync --locked --no-editable --no-install-package tflite-runtime --extra voice --extra wakeword --extra embedding --extra pi --extra test --python 3.12 --project $rootQuoted",
        "cd $rootQuoted"
    )
    if ($UseJarvisWakeWord) {
        $deployCommand += @(
            "sudo install -d -o orion-service -g orion-service -m 0750 $(Quote-Remote "$RemoteModelRoot/WAKEWORD")",
            "sudo install -o orion-service -g orion-service -m 0640 $(Quote-Remote "$remoteStage/models/hey_jarvis.onnx") $(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx")"
        )
    }
    $deployCommand += @(
        "sudo install -o root -g root -m 0644 $(Quote-Remote "$RemoteRoot/systemd/orion.service") $(Quote-Remote "/etc/systemd/system/$Service")",
        "printf %s $(Quote-Remote $modelEnvironmentBase64) | base64 -d | sudo python3 - $(Quote-Remote "/etc/orion/orion.env") $(Quote-Remote $remoteModel) $(Quote-Remote $legacyRemoteModel) $(Quote-Remote $RemoteLlamaServer) $(Quote-Remote $RemoteModelRoot)",
        "sudo -u orion-service env HOME=$testHomeQuoted ORION_DATABASE_PATH=$databaseQuoted PYTHONPATH=$(Quote-Remote "$RemoteRoot/src") $pythonQuoted -m pytest -p no:cacheprovider -q $(Quote-Remote "$RemoteRoot/tests")",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable $serviceQuoted"
    )
    if ($UseJarvisWakeWord) {
        $jarvisEnvironmentPython = @'
import os, pathlib, re, sys, tempfile
path = pathlib.Path(sys.argv[1])
values = {
    "ORION_VOICE_PROFILE": "legacy",
    "ORION_STT_BACKEND": "vosk",
    "ORION_WAKEWORD_MODEL_ID": "hey-jarvis",
    "ORION_AUDIO_DEVICE": "sysdefault",
    "ORION_WAKE_WORD_ENABLED": "true",
    "ORION_WAKE_WORD_MODEL_PATH": "__WAKEWORD_PATH__",
    "ORION_WAKE_WORD_MELSPECTROGRAM_PATH": "__MELSPECTROGRAM_PATH__",
    "ORION_WAKE_WORD_EMBEDDING_PATH": "__EMBEDDING_PATH__",
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
        $jarvisEnvironmentPython = $jarvisEnvironmentPython.Replace(
            "__WAKEWORD_PATH__",
            "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx"
        ).Replace(
            "__MELSPECTROGRAM_PATH__",
            "$RemoteRoot/artifacts/wakeword-staging/melspectrogram.onnx"
        ).Replace(
            "__EMBEDDING_PATH__",
            "$RemoteRoot/artifacts/wakeword-staging/embedding_model.onnx"
        )
        $jarvisEnvironmentBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($jarvisEnvironmentPython))
        $deployCommand += @(
            "printf %s $(Quote-Remote $jarvisEnvironmentBase64) | base64 -d | sudo python3 - $(Quote-Remote "/etc/orion/orion.env")",
            "sudo -u orion-service env HOME=/var/lib/orion ORION_VOICE_PROFILE=legacy ORION_STT_BACKEND=vosk ORION_AUDIO_DEVICE=sysdefault ORION_WAKE_WORD_ENABLED=true ORION_WAKE_WORD_MODEL_PATH=$(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx") ORION_WAKE_WORD_MELSPECTROGRAM_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/melspectrogram.onnx") ORION_WAKE_WORD_EMBEDDING_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/embedding_model.onnx") ORION_WAKE_WORD_THRESHOLD=$wakeWordThresholdValue ORION_WAKE_WORD_COOLDOWN_SECONDS=3.0 $pythonQuoted -m orion --validate-wakeword-assets",
            "sudo -u orion-service env HOME=/var/lib/orion ORION_VOICE_PROFILE=legacy ORION_STT_BACKEND=vosk ORION_AUDIO_DEVICE=sysdefault ORION_WAKE_WORD_ENABLED=true ORION_WAKE_WORD_MODEL_PATH=$(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx") ORION_WAKE_WORD_MELSPECTROGRAM_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/melspectrogram.onnx") ORION_WAKE_WORD_EMBEDDING_PATH=$(Quote-Remote "$RemoteRoot/artifacts/wakeword-staging/embedding_model.onnx") ORION_WAKE_WORD_THRESHOLD=$wakeWordThresholdValue ORION_WAKE_WORD_COOLDOWN_SECONDS=3.0 $pythonQuoted -c $(Quote-Remote "import numpy as np; from orion.audio.wakeword import OpenWakeWordDetector; from orion.config import Settings; d=OpenWakeWordDetector.from_settings(Settings.from_environment()); a=np.zeros(16000 * 3, dtype=np.int16); s=[d._model.predict(a[i:i+1280]) for i in range(0, len(a), 1280)]; assert len(s) == 38 and all(isinstance(x, dict) and x for x in s); print('Jarvis ONNX inference passed: 3 seconds, 38 frames')")"
        )
    }
    $deployCommand += @(
        "sudo systemctl start $serviceQuoted",
        "sudo systemctl is-active --quiet $serviceQuoted"
    )
    Invoke-Pi ($deployCommand -join " && ") "Deployment, locked environment sync, tests, and service start"
    $deploymentChanged = $true
    $serviceStopped = $false

    Write-Host "Registered Pi model inventory on Pi (see path and detail for local asset status):"
    try {
        $modelInventory = "sudo -u orion-service env HOME=/var/lib/orion ORION_MODEL_ROOT=$(Quote-Remote $RemoteModelRoot) ORION_STT_MODEL_ID=vosk-en-small ORION_VOSK_MODEL_PATH=$(Quote-Remote "$RemoteModelRoot/vosk-model-small-en-us-0.15") ORION_TTS_MODEL_ID=piper-lessac-medium ORION_WAKEWORD_MODEL_ID=hey-jarvis $(Quote-Remote "$remoteEnvironment/bin/orion-model") list"
        Invoke-Pi $modelInventory "Registered Pi model inventory"
    }
    catch {
        Write-Warning "Could not retrieve the Pi model inventory. Run '$remoteEnvironment/bin/orion-model list' on the Pi."
    }

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
            "sudo -u orion-service env HOME=/var/lib/orion UV_PYTHON_INSTALL_DIR=$(Quote-Remote $remotePythonInstall) UV_CACHE_DIR=/var/cache/orion/uv-cache UV_PROJECT_ENVIRONMENT=$(Quote-Remote $remoteEnvironment) $uvQuoted sync --locked --no-editable --no-install-package tflite-runtime --extra voice --extra wakeword --extra embedding --extra pi --extra test --python 3.12 --project $rootQuoted",
            "sudo systemctl daemon-reload"
        )
        if ($UseJarvisWakeWord) {
            $rollback += @(
                "if sudo test -f $backupQuoted/models/hey_jarvis.onnx; then sudo install -d -o orion-service -g orion-service -m 0750 $(Quote-Remote "$RemoteModelRoot/WAKEWORD") && sudo install -o orion-service -g orion-service -m 0640 $backupQuoted/models/hey_jarvis.onnx $(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx"); else sudo rm -f $(Quote-Remote "$RemoteModelRoot/WAKEWORD/hey_jarvis.onnx"); fi"
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
    if (Test-Path -LiteralPath $localModelStageParent) {
        Remove-Item -LiteralPath $localModelStageParent -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $script:PiSudoPassword) {
        try {
            Invoke-Pi "sudo rm -f $(Quote-Remote $temporaryDatabaseBackup)" "Temporary database backup cleanup"
        }
        catch {
            Write-Warning "Could not remove temporary database backup $temporaryDatabaseBackup."
        }
    }
    try {
        $cleanupCommand = "rm -rf $(Quote-Remote $remoteStage) $(Quote-Remote $remoteArchive) $(Quote-Remote $remoteModelStageParent) $(Quote-Remote $testHome) $(Quote-Remote $database) $(Quote-Remote "$database-wal") $(Quote-Remote "$database-shm")"
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