[CmdletBinding()]
param(
    [string]$WorkspacePath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\wakeword-training')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspace = [System.IO.Path]::GetFullPath($WorkspacePath)
$directories = @(
    'positive_train',
    'positive_test',
    'negative_train',
    'negative_test',
    'background_audio',
    'room_impulse_responses',
    'validation_features',
    'training_output'
)

New-Item -ItemType Directory -Path $workspace -Force | Out-Null
foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path (Join-Path $workspace $directory) -Force | Out-Null
}

$phraseFile = Join-Path $workspace 'target-phrase.txt'
if (-not (Test-Path -LiteralPath $phraseFile)) {
    Set-Content -LiteralPath $phraseFile -Value 'Hey Orion' -Encoding UTF8
}

Write-Output "Wake-word training workspace prepared at: $workspace"
Write-Output "Target phrase seed: $phraseFile"
Write-Output 'No models or datasets were downloaded. Keep private audio in this ignored workspace.'
Write-Output 'Use the pinned openWakeWord v0.6.0 automatic training notebook and review every dataset/model license.'
