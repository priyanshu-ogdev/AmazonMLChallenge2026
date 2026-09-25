<#
.SYNOPSIS
    Phase 5: Official Competition Submission Validation & Format Integrity Audit.
.DESCRIPTION
    Runs utils/validate_submission.py against matching_results.tsv and candidate_pairs.tsv:
    - Verifies TSV headers and delimiters
    - Checks for duplicate rows and duplicate candidate IDs
    - Validates source entity ID prefixes (S1, S2, S3)
    - Confirms all required test S1 entities are present
    - Verifies final matches are a strict subset of candidate_pairs.tsv
    - Optionally checks ID existence against test_source2.tsv and test_source3.tsv
.PARAMETER SubmissionDir
    Directory containing matching_results.tsv and candidate_pairs.tsv (default: output\phase4_submission).
.PARAMETER TestDir
    Path to dataset\test directory (default: dataset\test).
.PARAMETER CheckIds
    Perform deep validation checking that all matched IDs exist in test_source2/3.
.PARAMETER DryRun
    Display execution commands without executing them.
#>

param(
    [string]$SubmissionDir = "",
    [string]$TestDir = "",
    [switch]$CheckIds = $false,
    [switch]$DryRun = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "PHASE 5: COMPETITION SUBMISSION VALIDATION & INTEGRITY AUDIT"

$python = Get-PythonExecutable -ExplicitPath $PythonPath

if (-not $SubmissionDir) {
    $SubmissionDir = Join-Path $DEFAULT_OUT "phase4_submission"
}
if (-not $TestDir) {
    $TestDir = Join-Path $DATASET_DIR "test"
}

$matchingFile  = Join-Path $SubmissionDir "matching_results.tsv"
$candidateFile = Join-Path $SubmissionDir "candidate_pairs.tsv"

if ((-not (Test-Path $matchingFile)) -and (-not $DryRun)) {
    throw "matching_results.tsv not found in $SubmissionDir. Run Phase 4 first."
}

Write-Step "5.1" "Inspecting Submission Artifacts..."
Write-Info "Matching Results: $matchingFile"
if (Test-Path $candidateFile) {
    Write-Info "Candidate Pairs:  $candidateFile"
} else {
    Write-WarningMessage "candidate_pairs.tsv is missing from submission directory (recommended for packaging)."
}

Write-Step "5.2" "Executing Submission Validator..."
$valArgs = @(
    "utils/validate_submission.py",
    "--matching", $matchingFile,
    "--test-dir", $TestDir
)

if (Test-Path $candidateFile) {
    $valArgs += @("--candidate", $candidateFile)
}
if ($CheckIds) {
    $valArgs += "--check-ids"
}

Invoke-PythonScript "utils\validate_submission.py" $valArgs[1..($valArgs.Length-1)] "Submission Validator" -DryRun $DryRun -PythonExe $python

if (-not $DryRun) {
    Write-Host ""
    Write-Host "  ========================================================" -ForegroundColor Green
    Write-Host "  SUBMISSION IS FULLY COMPLIANT AND READY FOR LEADERBOARD" -ForegroundColor Green
    Write-Host "  ========================================================" -ForegroundColor Green
    Write-Host "  Package directory: $SubmissionDir" -ForegroundColor White
}

Write-Header "PHASE 5 COMPLETE"
