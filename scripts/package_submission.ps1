<#
.SYNOPSIS
    Package official competition submission archive (.zip).
.DESCRIPTION
    Validates matching_results.tsv and candidate_pairs.tsv with utils/validate_submission.py,
    then packs the official folder structure:
      <team_name>_submission.zip
      ├── output/
      │   ├── matching_results.tsv
      │   └── candidate_pairs.tsv
      ├── code/
      │   └── business_entity_resolution/
      │       ├── src/
      │       ├── README.md
      │       └── requirements.txt
      └── Documentation_template.md
.PARAMETER Matching
    Path to matching_results.tsv (required).
.PARAMETER Candidate
    Path to candidate_pairs.tsv (required).
.PARAMETER OutputZip
    Path to destination zip file (default: submissions\submission.zip).
.PARAMETER TestDir
    Path to test dataset directory for validation (default: dataset\test).
.PARAMETER SkipValidation
    Skip running utils/validate_submission.py.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Matching,

    [Parameter(Mandatory=$true)]
    [string]$Candidate,

    [string]$OutputZip = "submissions\submission.zip",
    [string]$TestDir = "",
    [switch]$SkipValidation = $false,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"

Write-Header "OFFICIAL COMPETITION SUBMISSION PACKAGING"

$python = Get-PythonExecutable -ExplicitPath $PythonPath
$script = Join-Path $PSScriptRoot "package_submission.py"

$pkgArgs = @(
    "--matching", (Resolve-FullPath $Matching),
    "--candidate", (Resolve-FullPath $Candidate),
    "--output-zip", (Resolve-FullPath $OutputZip)
)

if ($TestDir) {
    $pkgArgs += @("--test-dir", (Resolve-FullPath $TestDir))
}
if ($SkipValidation) {
    $pkgArgs += "--skip-validation"
}

Invoke-PythonScript $script $pkgArgs "Packaging Submission" -PythonExe $python

Write-Header "PACKAGING COMPLETE"
