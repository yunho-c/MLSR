[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ExecuTorchPrefix,
    [int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$qnnBuild = Join-Path $projectRoot 'target/qnn-native'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -property installationPath
& (Join-Path $vsRoot 'Common7/Tools/Launch-VsDevShell.ps1') -Arch arm64 -HostArch amd64 -SkipAutomaticLocation
& cmake -S (Join-Path $projectRoot 'native/qnn') -B $qnnBuild -G 'Visual Studio 17 2022' -A ARM64 "-DCMAKE_PREFIX_PATH=$ExecuTorchPrefix"
if ($LASTEXITCODE) { throw 'QNN bridge configure failed' }
& cmake --build $qnnBuild --config Release --parallel $Jobs
if ($LASTEXITCODE) { throw 'QNN bridge build failed' }
Write-Output "Set MLSR_QNN_BRIDGE_DIR to $qnnBuild/Release before building Cargo with --features qnn."
