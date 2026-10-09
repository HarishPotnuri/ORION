$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv-local\Scripts\python.exe"
$databaseId = [guid]::NewGuid().ToString("N")
$database = Join-Path ([System.IO.Path]::GetTempPath()) "orion-local-test-$databaseId.sqlite3"
$previousDatabasePath = $env:ORION_DATABASE_PATH

Push-Location $root
try {
    if (-not (Test-Path $python)) {
        py -3.11 -m venv .venv-local
    }

    & $python -m pip install -e ".[test]"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install ORION's local test profile."
    }

    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "ORION test suite failed."
    }

    $env:ORION_DATABASE_PATH = $database
    & $python -m orion --check-config
    if ($LASTEXITCODE -ne 0) {
        throw "ORION configuration check failed."
    }

    & (Join-Path $root ".venv-local\Scripts\orion-model.exe") current
    if ($LASTEXITCODE -ne 0) {
        throw "ORION model-selection CLI check failed."
    }

    & $python -m orion --request "What time is it?"
    if ($LASTEXITCODE -ne 0) {
        throw "Local time request failed."
    }

    & $python -m orion --request "Remind me tomorrow at 1 PM about ORION laptop test"
    if ($LASTEXITCODE -ne 0) {
        throw "Reminder creation request failed."
    }

    & $python -m orion --request "What reminders do I have today?"
    if ($LASTEXITCODE -ne 0) {
        throw "Reminder persistence request failed."
    }
}
finally {
    if ($null -eq $previousDatabasePath) {
        Remove-Item Env:ORION_DATABASE_PATH -ErrorAction SilentlyContinue
    }
    else {
        $env:ORION_DATABASE_PATH = $previousDatabasePath
    }
    Remove-Item $database, "$database-wal", "$database-shm" -ErrorAction SilentlyContinue
    Pop-Location
}