# Dot-source into the development shell. Uses installed local dependencies only.
[CmdletBinding()]
param(
    [string]$SdkRoot = 'C:/Qualcomm/AIStack/QAIRT/2.50.40.260831',
    [string]$BridgeDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) 'target/qnn-native/Release'),
    [string]$FastRpcDirectory
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath (Join-Path $BridgeDirectory 'mlsr_qnn_bridge.dll'))) {
    throw 'Build the QNN bridge with tools/build_qnn.ps1 first'
}
if (!(Test-Path -LiteralPath (Join-Path $SdkRoot 'lib/aarch64-windows-msvc/QnnHtp.dll'))) {
    throw 'The pinned QAIRT SDK must already be installed'
}
if (!$FastRpcDirectory) {
    $qnnDrivers = @(Get-ChildItem -Path 'C:/Windows/System32/DriverStore/FileRepository/qcnspmcdm8480.inf_arm64_*/libcdsprpc.dll')
    if ($qnnDrivers.Count -ne 1) { throw 'Specify -FastRpcDirectory for the installed X2 Elite driver' }
    $FastRpcDirectory = $qnnDrivers[0].DirectoryName
}
if (!(Test-Path -LiteralPath (Join-Path $FastRpcDirectory 'libcdsprpc.dll'))) { throw 'Missing ARM64 FastRPC driver' }
$env:MLSR_QNN_BRIDGE_DIR = (Resolve-Path -LiteralPath $BridgeDirectory).Path
$env:PATH = "$env:MLSR_QNN_BRIDGE_DIR;$(Join-Path $SdkRoot 'lib/aarch64-windows-msvc');$FastRpcDirectory;$env:PATH"
$env:ADSP_LIBRARY_PATH = Join-Path $SdkRoot 'lib/hexagon-v81/unsigned'
