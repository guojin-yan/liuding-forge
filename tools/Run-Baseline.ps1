[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [switch]$ReuseBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Baseline output is intentionally outside the source tree's tracked paths.
# The report keeps command timing and benchmark summaries; raw logs stay next
# to the run and can be deleted after the evidence summary is recorded.
$sourceRoot = (Resolve-Path $SourceDir).Path
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $sourceRoot "out/build/baseline-release"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $sourceRoot "out/baseline/release"
}
$buildContainer = Join-Path $sourceRoot "out/build"
$outputContainer = Join-Path $sourceRoot "out/baseline"
New-Item -ItemType Directory -Path $buildContainer -Force | Out-Null
New-Item -ItemType Directory -Path $outputContainer -Force | Out-Null

function Assert-ContainedPath {
    param([string]$Path, [string]$Root, [string]$Label)
    $resolvedPath = if (Test-Path $Path) {
        (Resolve-Path $Path).Path
    } else {
        [IO.Path]::GetFullPath($Path)
    }
    $resolvedRoot = (Resolve-Path $Root).Path
    $resolvedPath = $resolvedPath.TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedRoot = $resolvedRoot.TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $resolvedPath.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must remain under ${resolvedRoot}: $resolvedPath"
    }
}

Assert-ContainedPath $BuildDir $buildContainer "BuildDir"
Assert-ContainedPath $OutputDir $outputContainer "OutputDir"

if (-not $ReuseBuild -and (Test-Path $BuildDir)) {
    [IO.Directory]::Delete([IO.Path]::GetFullPath($BuildDir), $true)
}
if (Test-Path $OutputDir) {
    [IO.Directory]::Delete([IO.Path]::GetFullPath($OutputDir), $true)
}
$outputRoot = (New-Item -ItemType Directory -Path $OutputDir -Force).FullName

function Invoke-Captured {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogName
    )
    $logPath = Join-Path $outputRoot $LogName
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    & $Name @Arguments 2>&1 |
        Tee-Object -FilePath $logPath |
        ForEach-Object { Write-Host $_ }
    $exitCode = $LASTEXITCODE
    $stopwatch.Stop()
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode; see $logPath"
    }
    return [ordered]@{
        command = "$Name $($Arguments -join ' ')"
        exit_code = $exitCode
        elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        log = $LogName
    }
}

$commit = (& git -C $sourceRoot rev-parse HEAD).Trim()
$dirty = @(& git -C $sourceRoot status --porcelain).Count -ne 0
$cmakeVersion = (& cmake --version | Select-Object -First 1).Trim()
$steps = [Collections.Generic.List[object]]::new()

$steps.Add((Invoke-Captured cmake @(
        "-S", $sourceRoot,
        "-B", $BuildDir,
        "-DJYPPX_BUILD_TESTS=ON",
        "-DJYPPX_BUILD_BENCHMARKS=ON",
        "-DJYPPX_BUILD_EXAMPLES=OFF",
        "-DJYPPX_BUILD_DOCS=OFF",
        "-DCMAKE_PREFIX_PATH=$(Join-Path $sourceRoot 'third_party/install/libtorch')"
    ) "01-configure.log"))
$steps.Add((Invoke-Captured cmake @(
        "--build", $BuildDir, "--config", "Release", "--parallel"
    ) "02-build.log"))
$steps.Add((Invoke-Captured ctest @(
        "--test-dir", $BuildDir, "-C", "Release", "--output-on-failure"
    ) "03-ctest.log"))

$runningOnWindows = [Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
    [Runtime.InteropServices.OSPlatform]::Windows)
$benchmarkDir = if ($runningOnWindows) {
    Join-Path $BuildDir "tests/benchmarks/Release"
} else {
    Join-Path $BuildDir "tests/benchmarks"
}
$executableSuffix = if ($runningOnWindows) { ".exe" } else { "" }
$benchmarkNames = @(
    "jyppx_bench_dataloader",
    "jyppx_bench_optimizer_convergence",
    "jyppx_bench_trainer_throughput"
)
$benchmarks = [ordered]@{}
foreach ($benchmarkName in $benchmarkNames) {
    $executable = Join-Path $benchmarkDir "$benchmarkName$executableSuffix"
    if (-not (Test-Path $executable)) {
        throw "Expected benchmark executable is missing: $executable"
    }
    $logName = "benchmark-$benchmarkName.csv"
    $benchmarks[$benchmarkName] = Invoke-Captured $executable @() $logName
}

$memoryRuns = [Collections.Generic.List[object]]::new()
$memoryRows = [Collections.Generic.List[object]]::new()
foreach ($workers in @(0, 1, 4, 8)) {
    $executable = Join-Path $benchmarkDir "jyppx_bench_memory_usage$executableSuffix"
    $logName = "benchmark-jyppx_bench_memory_usage-$workers.csv"
    $memoryRuns.Add((Invoke-Captured $executable @("--workers", "$workers") $logName))
    foreach ($row in (Import-Csv (Join-Path $outputRoot $logName))) {
        $memoryRows.Add($row)
    }
}
$benchmarks["jyppx_bench_memory_usage"] = [ordered]@{
    isolation = "one process per worker configuration"
    runs = $memoryRuns
}
$peakValues = @($memoryRows | ForEach-Object { [UInt64]$_.peak_bytes })
$maxPeakBytes = ($peakValues | Measure-Object -Maximum).Maximum

$artifactNames = @(
    "jyppx_foundation.lib",
    "jyppx_training_framework.lib",
    "libjyppx_foundation.a",
    "libjyppx_training_framework.a",
    "jyppx_bench_dataloader$executableSuffix",
    "jyppx_bench_trainer_throughput$executableSuffix",
    "jyppx_bench_optimizer_convergence$executableSuffix",
    "jyppx_bench_memory_usage$executableSuffix"
)
$artifacts = @(
    Get-ChildItem $BuildDir -Recurse -File |
        Where-Object { $_.Name -in $artifactNames } |
        ForEach-Object {
            [ordered]@{
                path = [IO.Path]::GetRelativePath($BuildDir, $_.FullName)
                bytes = $_.Length
            }
        }
)

$report = [ordered]@{
    schema_version = 1
    kind = "jyppx-local-baseline"
    commit = $commit
    dirty = $dirty
    configuration = "Release"
    cmake = $cmakeVersion
    environment = [ordered]@{
        os = [Runtime.InteropServices.RuntimeInformation]::OSDescription
        architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        logical_processors = [Environment]::ProcessorCount
        processor = $env:PROCESSOR_IDENTIFIER
        libtorch_build = (Get-Content (Join-Path $sourceRoot "third_party/install/libtorch/build-version") -Raw).Trim()
    }
    source_dir = $sourceRoot
    steps = $steps
    artifacts = $artifacts
    benchmarks = $benchmarks
    memory = [ordered]@{
        source = "jyppx_bench_memory_usage"
        peak_working_set_bytes = [UInt64]$maxPeakBytes
        rows = $memoryRows
    }
}
$reportPath = Join-Path $outputRoot "baseline.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $reportPath -Encoding utf8
Write-Output "Baseline report: $reportPath"
