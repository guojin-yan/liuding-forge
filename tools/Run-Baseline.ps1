# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0 (the "License").

[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$LibTorchDir = "",
    [ValidateRange(3, 20)]
    [int]$Samples = 5,
    [ValidateRange(0, 10)]
    [int]$Warmup = 1,
    [ValidateRange(0.01, 100.0)]
    [double]$MaxCvPercent = 10.0,
    [ValidateRange(0.01, 100.0)]
    [double]$MinRegressionPercent = 10.0,
    [string]$ReferenceFile = "",
    [switch]$FreshBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Baseline output is intentionally outside tracked paths. The report retains
# raw samples and statistics; command logs can be deleted after the evidence
# summary and reference threshold are recorded.
$sourceRoot = (Resolve-Path $SourceDir).Path
if ([string]::IsNullOrWhiteSpace($LibTorchDir)) {
    $LibTorchDir = Join-Path $sourceRoot "third_party/install/libtorch"
}
if (-not (Test-Path -LiteralPath $LibTorchDir -PathType Container)) {
    throw "LibTorchDir does not exist: $LibTorchDir"
}
$libTorchRoot = (Resolve-Path -LiteralPath $LibTorchDir).Path
$torchConfig = Join-Path $libTorchRoot "share/cmake/Torch/TorchConfig.cmake"
if (-not (Test-Path -LiteralPath $torchConfig -PathType Leaf)) {
    throw "LibTorchDir does not contain TorchConfig.cmake: $libTorchRoot"
}
$libTorchBuildFile = Join-Path $libTorchRoot "build-version"
if (-not (Test-Path -LiteralPath $libTorchBuildFile -PathType Leaf)) {
    throw "LibTorchDir does not contain build-version: $libTorchRoot"
}
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
    if (-not $resolvedPath.StartsWith(
            $resolvedRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must remain under ${resolvedRoot}: $resolvedPath"
    }
}

Assert-ContainedPath $BuildDir $buildContainer "BuildDir"
Assert-ContainedPath $OutputDir $outputContainer "OutputDir"

if ($FreshBuild -and (Test-Path $BuildDir)) {
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

$metricDefinitions = [ordered]@{}
$metricSamples = [ordered]@{}
function Add-MetricSample {
    param(
        [string]$Name,
        [ValidateSet("higher", "lower")]
        [string]$Direction,
        [string]$Unit,
        [double]$Value
    )
    if (-not [double]::IsFinite($Value) -or $Value -le 0.0) {
        throw "$Name produced a non-positive or non-finite sample: $Value"
    }
    if (-not $script:metricDefinitions.Contains($Name)) {
        $script:metricDefinitions[$Name] = [ordered]@{
            direction = $Direction
            unit = $Unit
        }
        $script:metricSamples[$Name] = [Collections.Generic.List[double]]::new()
    }
    $definition = $script:metricDefinitions[$Name]
    if ($definition.direction -ne $Direction -or $definition.unit -ne $Unit) {
        throw "$Name changed direction or unit within one baseline run"
    }
    $script:metricSamples[$Name].Add($Value)
}

$commit = (& git -C $sourceRoot rev-parse HEAD).Trim()
$dirty = @(& git -C $sourceRoot status --porcelain).Count -ne 0
$cmakeVersion = (& cmake --version | Select-Object -First 1).Trim()
$steps = [Collections.Generic.List[object]]::new()
$benchmarkRuns = [Collections.Generic.List[object]]::new()
$dataPipelineObservations = [Collections.Generic.List[object]]::new()

$steps.Add((Invoke-Captured cmake @(
        "-S", $sourceRoot,
        "-B", $BuildDir,
        "-DJYPPX_BUILD_TESTS=ON",
        "-DJYPPX_BUILD_BENCHMARKS=ON",
        "-DJYPPX_BUILD_EXAMPLES=OFF",
        "-DJYPPX_BUILD_DOCS=OFF",
        "-DCMAKE_PREFIX_PATH=$libTorchRoot"
    ) "01-configure.log"))
$pythonCacheEntry = Get-Content (Join-Path $BuildDir "CMakeCache.txt") |
    Where-Object { $_ -match '^_?Python3_EXECUTABLE:(?:FILEPATH|INTERNAL)=(.+)$' } |
    Select-Object -First 1
if ($null -eq $pythonCacheEntry -or
        $pythonCacheEntry -notmatch '^_?Python3_EXECUTABLE:(?:FILEPATH|INTERNAL)=(.+)$') {
    throw "CMake did not resolve a Python3 interpreter"
}
$pythonCommand = $Matches[1]
if (-not (Test-Path -LiteralPath $pythonCommand -PathType Leaf)) {
    throw "CMake resolved a missing Python3 interpreter: $pythonCommand"
}
$compilerFile = Get-ChildItem (Join-Path $BuildDir "CMakeFiles") -Recurse `
    -File -Filter "CMakeCXXCompiler.cmake" | Select-Object -First 1
if ($null -eq $compilerFile) {
    throw "CMake did not write CMakeCXXCompiler.cmake"
}
$compilerText = Get-Content -LiteralPath $compilerFile.FullName -Raw
$compilerIdMatch = [regex]::Match(
    $compilerText, 'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)')
$compilerVersionMatch = [regex]::Match(
    $compilerText, 'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)')
if (-not $compilerIdMatch.Success -or -not $compilerVersionMatch.Success) {
    throw "CMake compiler identity is incomplete: $($compilerFile.FullName)"
}
$compilerIdentity = "$($compilerIdMatch.Groups[1].Value)-$($compilerVersionMatch.Groups[1].Value)"
$cacheText = Get-Content (Join-Path $BuildDir "CMakeCache.txt") -Raw
$generatorMatch = [regex]::Match(
    $cacheText, '(?m)^CMAKE_GENERATOR:INTERNAL=(.+)$')
if (-not $generatorMatch.Success) {
    throw "CMake generator identity is missing"
}
$generatorIdentity = $generatorMatch.Groups[1].Value.Trim()
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
$dataPipelineExecutable = Join-Path $benchmarkDir "jyppx_bench_data_pipeline$executableSuffix"
$trainerExecutable = Join-Path $benchmarkDir "jyppx_bench_trainer_throughput$executableSuffix"
$optimizerExecutable = Join-Path $benchmarkDir "jyppx_bench_optimizer_convergence$executableSuffix"
$adadeltaExecutable = Join-Path $benchmarkDir "jyppx_bench_adadelta_native_throughput$executableSuffix"
$nadamExecutable = Join-Path $benchmarkDir "jyppx_bench_nadam_native_throughput$executableSuffix"
$memoryExecutable = Join-Path $benchmarkDir "jyppx_bench_memory_usage$executableSuffix"
foreach ($executable in @(
        $dataPipelineExecutable,
        $trainerExecutable,
        $optimizerExecutable,
        $adadeltaExecutable,
        $nadamExecutable,
        $memoryExecutable)) {
    if (-not (Test-Path $executable -PathType Leaf)) {
        throw "Expected benchmark executable is missing: $executable"
    }
}

# Warmups execute in separate processes and are retained as logs, but do not
# enter the statistics. This avoids silently mixing cold-start and steady-state
# samples while still making warmup behavior auditable.
for ($index = 1; $index -le $Warmup; ++$index) {
    foreach ($workers in @(0, 1, 2, 4, 8)) {
        $benchmarkRuns.Add((Invoke-Captured -Name $dataPipelineExecutable `
                -Arguments @("--repeats", "1", "--workers", "$workers") `
                -LogName "warmup-data-pipeline-$workers-$index.csv"))
    }
    $benchmarkRuns.Add((Invoke-Captured -Name $trainerExecutable `
            -Arguments @() -LogName "warmup-trainer-$index.csv"))
    foreach ($workers in @(0, 1, 4, 8)) {
        $benchmarkRuns.Add((Invoke-Captured -Name $memoryExecutable `
                -Arguments @("--workers", "$workers") `
                -LogName "warmup-memory-$workers-$index.csv"))
    }
}

for ($sample = 1; $sample -le $Samples; ++$sample) {
    foreach ($workers in @(0, 1, 2, 4, 8)) {
        $dataLog = "sample-data-pipeline-$workers-$sample.csv"
        $benchmarkRuns.Add((Invoke-Captured -Name $dataPipelineExecutable `
                -Arguments @("--repeats", "1", "--workers", "$workers") `
                -LogName $dataLog))
        $dataRows = @(Import-Csv (Join-Path $outputRoot $dataLog))
        if ($dataRows.Count -ne 1) {
            throw "DataPipeline sample $sample for $workers workers must contain one row"
        }
        $row = $dataRows[0]
        Add-MetricSample "data_pipeline.elapsed_ms.workers_$($row.workers)" "lower" `
            "milliseconds" ([double]$row.elapsed_ms)
        Add-MetricSample "data_pipeline.samples_per_second.workers_$($row.workers)" `
            "higher" "samples_per_second" ([double]$row.samples_per_second)
        Add-MetricSample "data_pipeline.p95_batch_latency_ms.workers_$($row.workers)" `
            "lower" "milliseconds" ([double]$row.p95_batch_latency_ms)
        Add-MetricSample "data_pipeline.peak_resident_bytes.workers_$($row.workers)" `
            "lower" "bytes" ([double]$row.peak_resident_bytes)
        if ($workers -gt 0) {
            Add-MetricSample "data_pipeline.worker_utilization.workers_$($row.workers)" `
                "higher" "ratio" ([double]$row.worker_utilization)
        }
        $dataPipelineObservations.Add([ordered]@{
                sample = $sample
                workers = $workers
                mean_outstanding_batches = [double]$row.mean_outstanding_batches
                peak_outstanding_batches = [int]$row.peak_outstanding_batches
                peak_active_workers = [int]$row.peak_active_workers
                resident_delta_bytes = [double]$row.resident_delta_bytes
            })
    }

    $trainerLog = "sample-trainer-$sample.csv"
    $benchmarkRuns.Add((Invoke-Captured $trainerExecutable @() $trainerLog))
    $trainerRows = @(Import-Csv (Join-Path $outputRoot $trainerLog))
    if ($trainerRows.Count -ne 4) {
        throw "Trainer sample $sample must contain exactly four batch-size rows"
    }
    foreach ($row in $trainerRows) {
        Add-MetricSample "trainer.samples_per_second.batch_$($row.batch_size)" "higher" `
            "samples_per_second" ([double]$row.samples_per_second)
    }

    foreach ($workers in @(0, 1, 4, 8)) {
        $memoryLog = "sample-memory-$workers-$sample.csv"
        $benchmarkRuns.Add((Invoke-Captured -Name $memoryExecutable `
                -Arguments @("--workers", "$workers") -LogName $memoryLog))
        $memoryRows = @(Import-Csv (Join-Path $outputRoot $memoryLog))
        if ($memoryRows.Count -ne 1) {
            throw "Memory sample $sample for $workers workers must contain one row"
        }
        Add-MetricSample "memory.peak_bytes.workers_$workers" "lower" "bytes" `
            ([double]$memoryRows[0].peak_bytes)
    }
}

# Optimizer convergence is deterministic correctness evidence rather than a
# timing metric, so one complete trajectory is retained without aggregation.
$benchmarkRuns.Add((Invoke-Captured -Name $optimizerExecutable -Arguments @() `
        -LogName "benchmark-optimizer-convergence.csv"))
$benchmarkRuns.Add((Invoke-Captured -Name $adadeltaExecutable -Arguments @() `
        -LogName "benchmark-adadelta-native-throughput.csv"))
$benchmarkRuns.Add((Invoke-Captured -Name $nadamExecutable -Arguments @() `
        -LogName "benchmark-nadam-native-throughput.csv"))

$libtorchBuild = (Get-Content -LiteralPath $libTorchBuildFile -Raw).Trim()
$osDescription = [Runtime.InteropServices.RuntimeInformation]::OSDescription
$architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
$processor = if ([string]::IsNullOrWhiteSpace($env:PROCESSOR_IDENTIFIER)) {
    "unknown-processor"
} else {
    $env:PROCESSOR_IDENTIFIER.Trim()
}
$environmentKey = @(
    $osDescription,
    $architecture,
    $processor,
    $generatorIdentity,
    $compilerIdentity,
    "libtorch-$libtorchBuild",
    "Release"
) -join "|"

$metricRows = @(
    foreach ($name in $metricDefinitions.Keys) {
        [ordered]@{
            name = $name
            direction = $metricDefinitions[$name].direction
            unit = $metricDefinitions[$name].unit
            samples = @($metricSamples[$name])
        }
    }
)
$rawSampleReport = [ordered]@{
    schema_version = 1
    environment_key = $environmentKey
    metadata = [ordered]@{
        commit = $commit
        dirty = $dirty
        warmup_processes = $Warmup
        sample_processes = $Samples
    }
    data_pipeline_observations = $dataPipelineObservations
    metrics = $metricRows
}
$rawSamplePath = Join-Path $outputRoot "samples.json"
$rawSampleReport | ConvertTo-Json -Depth 8 |
    Set-Content -Path $rawSamplePath -Encoding utf8

$statisticsPath = Join-Path $outputRoot "statistics.json"
$statisticsArguments = @(
    (Join-Path $sourceRoot "tools/baseline_stats.py"),
    $rawSamplePath,
    "--output", $statisticsPath,
    "--max-cv-percent", "$MaxCvPercent",
    "--min-regression-percent", "$MinRegressionPercent"
)
if (-not [string]::IsNullOrWhiteSpace($ReferenceFile)) {
    $resolvedReference = (Resolve-Path -LiteralPath $ReferenceFile).Path
    $statisticsArguments += @("--reference", $resolvedReference)
}
$steps.Add((Invoke-Captured $pythonCommand $statisticsArguments "04-statistics.log"))
$statistics = Get-Content $statisticsPath -Raw | ConvertFrom-Json -AsHashtable

$libraryDir = if ($runningOnWindows) {
    Join-Path $BuildDir "Release"
} else {
    $BuildDir
}
$libraryNames = if ($runningOnWindows) {
    @(
        "jyppx_foundation.lib",
        "jyppx_core.lib",
        "jyppx_interop.lib",
        "jyppx_data.lib",
        "jyppx_nn.lib",
        "jyppx_optim.lib",
        "jyppx_training.lib",
        "jyppx_experiment.lib"
    )
} else {
    @(
        "libjyppx_foundation.a",
        "libjyppx_core.a",
        "libjyppx_interop.a",
        "libjyppx_data.a",
        "libjyppx_nn.a",
        "libjyppx_optim.a",
        "libjyppx_training.a",
        "libjyppx_experiment.a"
    )
}
$artifactPaths = @(
    $libraryNames | ForEach-Object { Join-Path $libraryDir $_ }
    @(
        $dataPipelineExecutable,
        $trainerExecutable,
        $optimizerExecutable,
        $memoryExecutable
    )
)
$artifacts = @(
    foreach ($artifactPath in $artifactPaths) {
        if (-not (Test-Path $artifactPath -PathType Leaf)) {
            throw "Expected baseline artifact is missing: $artifactPath"
        }
        $artifact = Get-Item $artifactPath
        [ordered]@{
            path = [IO.Path]::GetRelativePath($BuildDir, $artifact.FullName)
            bytes = $artifact.Length
        }
    }
)

$report = [ordered]@{
    schema_version = 2
    kind = "jyppx-local-baseline"
    commit = $commit
    dirty = $dirty
    configuration = "Release"
    cmake = $cmakeVersion
    environment = [ordered]@{
        key = $environmentKey
        os = $osDescription
        architecture = $architecture
        logical_processors = [Environment]::ProcessorCount
        processor = $processor
        generator = $generatorIdentity
        compiler = $compilerIdentity
        libtorch_build = $libtorchBuild
    }
    sampling = [ordered]@{
        warmup_processes = $Warmup
        sample_processes = $Samples
        max_cv_percent = $MaxCvPercent
        min_regression_percent = $MinRegressionPercent
        reference = if ([string]::IsNullOrWhiteSpace($ReferenceFile)) {
            $null
        } else {
            (Resolve-Path -LiteralPath $ReferenceFile).Path
        }
    }
    source_dir = $sourceRoot
    steps = $steps
    benchmark_runs = $benchmarkRuns
    data_pipeline_observations = $dataPipelineObservations
    artifacts = $artifacts
    statistics = $statistics
}
$reportPath = Join-Path $outputRoot "baseline.json"
$report | ConvertTo-Json -Depth 12 | Set-Content -Path $reportPath -Encoding utf8
Write-Output "Baseline report: $reportPath"
