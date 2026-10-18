# Explicit Windows ARM64 real-model checks. No model or SDK downloads.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Go,
    [Parameter(Mandatory = $true)][string]$Dart,
    [Parameter(Mandatory = $true)][string]$CCompiler,
    [string]$SdkRoot = 'C:/Qualcomm/AIStack/QAIRT/2.50.40.260831'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$bundlePath = (Resolve-Path -LiteralPath $Bundle).Path
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputPath | Out-Null
$manifest = Join-Path $projectRoot 'model-manifests/supertonic3-qnn-superfast.yaml'
$catalog = Join-Path $projectRoot 'model-manifests/catalog-qnn-experimental.yaml'
$native = Join-Path $projectRoot 'target/release'
$report = [ordered]@{ passed = $false; host = 'Windows ARM64'; manifest_sha256 = (Get-FileHash $manifest).Hash.ToLower(); checks = @() }
$report.experimental_catalog_sha256 = (Get-FileHash $catalog).Hash.ToLower()

function Invoke-Check([string]$Name, [string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $outputPath "$Name.log")
    if ($LASTEXITCODE) { throw "$Name failed with exit code $LASTEXITCODE" }
    $report.checks += $Name
}
function Quote-GoFlag([string]$Value) {
    $value = $Value.Replace('\', '/')
    if ($value -match '\s') { return '"' + $value + '"' }
    return $value
}

Push-Location $projectRoot
try {
    . (Join-Path $PSScriptRoot 'qnn_env.ps1') -SdkRoot $SdkRoot
    $report.bridge_sha256 = (Get-FileHash (Join-Path $env:MLSR_QNN_BRIDGE_DIR 'mlsr_qnn_bridge.dll')).Hash.ToLower()
    Invoke-Check 'rust-build' 'cargo' @('build', '--locked', '--release', '-p', 'mlsr-ffi', '--features', 'qnn')
    $env:PATH = "$native;$(Split-Path -Parent $CCompiler);$env:PATH"
    $env:MLSR_LIBRARY_PATH = Join-Path $native 'mlsr_ffi.dll'
    $env:MLSR_TEST_MANIFEST = $manifest
    $env:MLSR_TEST_BUNDLE = $bundlePath
    $env:MLSR_TEST_POLICY = 'qnn_htp'
    $env:MLSR_TEST_CATALOG = $catalog
    $env:MLSR_TEST_VERIFIED_BUNDLE = $bundlePath
    $env:MLSR_TEST_DELIVERY_BUNDLE = $bundlePath
    $env:MLSR_TEST_DELIVERY_CATALOG = $catalog
    $env:MLSR_TEST_DELIVERY_ID = 'supertonic3-qnn-superfast-v1'
    $env:MLSR_TEST_DELIVERY_PLATFORM = 'windows-arm64'
    Remove-Item Env:MLSR_TEST_NATIVE_OUTPUT -ErrorAction SilentlyContinue
    $env:PYTHONPATH = Join-Path $projectRoot 'bindings/python/src'
    $report.native_sha256 = (Get-FileHash $env:MLSR_LIBRARY_PATH).Hash.ToLower()
    $report.versions = [ordered]@{
        python = (& $Python -c 'import sys,platform; print(sys.version,platform.machine())' | Out-String).Trim()
        go = (& $Go version | Out-String).Trim()
        dart = (& $Dart --version 2>&1 | Out-String).Trim()
        cc = (& $CCompiler --version | Out-String).Trim()
    }
    $cBinary = Join-Path $outputPath 'c-smoke.exe'
    Invoke-Check 'c-build' $CCompiler @('-std=c11', '-Wall', '-Wextra', '-Werror', "-I$projectRoot/include", 'tests/integration/synthesis_smoke.c', "-L$native", '-l:mlsr_ffi.dll.lib', '-o', $cBinary)
    Invoke-Check 'c-smoke' $cBinary @($manifest, $bundlePath, 'qnn_htp')
    $cSdkBinary = Join-Path $outputPath 'c-sdk-smoke.exe'
    $cSdkOutput = Join-Path $outputPath 'c-sdk'
    New-Item -ItemType Directory -Path $cSdkOutput | Out-Null
    Invoke-Check 'c-sdk-build' $CCompiler @('-std=c11', '-Wall', '-Wextra', '-Werror', "-I$projectRoot/include", 'tests/integration/developer_smoke.c', "-L$native", '-l:mlsr_ffi.dll.lib', '-o', $cSdkBinary)
    Invoke-Check 'c-sdk' $cSdkBinary @($manifest, $bundlePath, $cSdkOutput, 'qnn_htp')
    $env:MLSR_TEST_OUTPUT = Join-Path $outputPath 'python-pcm'
    $env:MLSR_SDK_OUTPUT = Join-Path $outputPath 'python-sdk'
    Invoke-Check 'python' $Python @('-m', 'unittest', 'discover', '-s', 'bindings/python/tests', '-v')
    Invoke-Check 'python-sdk-example' $Python @('bindings/python/examples/developer_api.py', '--manifest', $manifest, '--bundle', $bundlePath, '--policy', 'qnn_htp', '--cancel-first', '--output', (Join-Path $outputPath 'python-sdk-example.wav'))
    $env:CC = Quote-GoFlag $CCompiler
    $env:CGO_ENABLED = '1'
    $env:CGO_CFLAGS = Quote-GoFlag "-I$projectRoot/include"
    $env:CGO_LDFLAGS = Quote-GoFlag "-L$native"
    $env:MLSR_TEST_OUTPUT = Join-Path $outputPath 'go-pcm'
    $env:MLSR_SDK_OUTPUT = Join-Path $outputPath 'go-sdk'
    Push-Location (Join-Path $projectRoot 'bindings/go')
    try {
        Invoke-Check 'go' $Go @('test', '-count=1', '-timeout=120s', '-v', './...')
        Invoke-Check 'go-vet' $Go @('vet', './...')
        Invoke-Check 'go-sdk-example' $Go @('run', './examples/developer_api', '--manifest', $manifest, '--bundle', $bundlePath, '--policy', 'qnn_htp', '--output', (Join-Path $outputPath 'go-sdk-example.wav'))
    } finally { Pop-Location }

    $dartRoot = Join-Path $outputPath 'dart-project'
    New-Item -ItemType Directory -Path (Join-Path $dartRoot 'test') | Out-Null
    foreach ($file in @('synthesis_test.dart', 'sdk_test.dart', 'model_store_cases.dart', 'model_fixture.dart')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot "bindings/dart/test/$file") -Destination (Join-Path $dartRoot "test/$file")
    }
    Copy-Item -LiteralPath (Join-Path $projectRoot 'bindings/dart/test/fixtures') -Destination (Join-Path $dartRoot 'test/fixtures') -Recurse
    Copy-Item -LiteralPath (Join-Path $projectRoot 'bindings/dart/example/developer_api.dart') -Destination (Join-Path $dartRoot 'developer_api.dart')
    # JSON is valid YAML and preserves arbitrary Windows paths without shell
    # interpolation. A consuming package exercises the actual native-asset hook.
    @{
        name = 'mlsr_qnn_binding_smoke'
        environment = @{ sdk = '>=3.11.0 <4.0.0' }
        dependencies = @{ mlsr = @{ path = (Join-Path $projectRoot 'bindings/dart') } }
        dev_dependencies = @{ test = '^1.33.0'; crypto = '^3.0.7'; yaml = '^3.1.4' }
        hooks = @{ user_defines = @{ mlsr = @{ qnn = $true } } }
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $dartRoot 'pubspec.yaml') -Encoding utf8
    $env:MLSR_TEST_OUTPUT = Join-Path $outputPath 'dart-pcm'
    $env:MLSR_SDK_OUTPUT = Join-Path $outputPath 'dart-sdk'
    Push-Location $dartRoot
    try {
        Invoke-Check 'dart-get' $Dart @('pub', 'get')
        Invoke-Check 'dart' $Dart @('test', '--concurrency=1', '-r', 'expanded')
        Invoke-Check 'dart-sdk-example' $Dart @('run', 'developer_api.dart', $manifest, $bundlePath, (Join-Path $outputPath 'dart-sdk-example.wav'), 'qnn_htp')
        $report.leased_model_engine_passed = $true
    } finally { Pop-Location }
    $report.pcm = @()
    foreach ($name in @('en_M1', 'en_F1', 'ko_M1', 'ko_F1')) {
        $files = @('python', 'go', 'dart') | ForEach-Object { Join-Path $outputPath "$_-pcm/$name.f32" }
        $hashes = @($files | ForEach-Object { (Get-FileHash -LiteralPath $_).Hash.ToLower() })
        if (@($hashes | Select-Object -Unique).Count -ne 1) { throw "Binding PCM differs: $name" }
        $report.pcm += @{ case = $name; frames = (Get-Item -LiteralPath $files[0]).Length / 4; sha256 = $hashes[0]; byte_exact_across_bindings = $true }
    }
    Invoke-Check 'sdk-pcm' $Python @('tools/check_qnn_sdk_pcm.py', $outputPath)
    $report.developer_sdk = Get-Content -LiteralPath (Join-Path $outputPath 'sdk-pcm.json') -Raw | ConvertFrom-Json
    $report.headless_sdk_passed = $true
    $report.native_windows_playback = 'unsupported_capability'
    $report.passed = $true
} catch {
    $report.failure = $_.Exception.Message
    throw
} finally {
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputPath 'report.json') -Encoding utf8
    Pop-Location
}
