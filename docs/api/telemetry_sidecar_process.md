# Telemetry sidecar process transport

`jyppx::liuding::cli::SidecarProcessTransport` is the product CLI's implementation of
`experiment::TelemetrySidecarTransport`. It launches a fixed executable and
exchanges bounded UTF-8 JSON lines with that process. It is private tool code;
it does not add subprocess management to the installed foundation API.

## Launch and ownership

Supply an absolute executable path, UTF-8 arguments, an existing isolated working
directory, and explicit child environment configuration. The caller reserves the
directory for this observer and serializes all calls. Relative executables and
PATH lookup are rejected. Arguments are passed directly to the executable and
never interpreted by a shell. Child environment overrides do not modify the
parent's environment or current directory.

```cpp
jyppx::liuding::cli::SidecarProcessOptions process;
process.executable = locked_python_executable;       // absolute native path
process.working_directory = exclusive_sidecar_cwd;   // existing private directory
process.arguments = {
    "-I", "-u", sidecar_script_utf8,
    "--backend", "tensorboard",
    "--tensorboard-output", new_output_directory_utf8};
process.environment = locked_child_environment;
auto channel = std::make_shared<jyppx::liuding::cli::SidecarProcessTransport>(process);
jyppx::liuding::experiment::OfficialTelemetrySink observer(channel);
// Use observer.emit/flush/close with the operation's absolute deadline.
```

Environment inheritance is disabled by default. A locked child environment must
include any platform runtime prerequisites, such as the required Windows system
environment. Set `inherit_environment=true` explicitly when a snapshot of the
parent environment is intended; overrides then apply only to the child. The
application must not concurrently mutate its environment during that snapshot.

The telemetry sidecar and dependency files are user-facing product tools under
`tools/telemetry_sidecar`. Its fixed backend/output/collector options are launch
configuration. Never derive an executable, arguments, environment, cwd or output
destination from an event payload. Pin and verify the actual runtime separately;
successfully launching a Python executable does not verify its packages.

## Framing and limits

`transact` appends one LF to a nonempty UTF-8 request. Embedded NUL, CR or LF is
rejected. It returns one nonempty UTF-8 reply without LF. The default request and
reply bounds are 16 MiB and 128 KiB. The underlying pipes have finite kernel
buffers; the implementation drains stdout and stderr while it writes stdin,
including when the child stops reading a large request.

The child must remain alive between exchanges. EOF, a partial line, process exit,
invalid UTF-8, embedded NUL/CR, or extra bytes after the reply LF fail explicitly.
Available unsolicited stdout is rejected before the next request. A short
configurable quiet period (2 ms by default) drains immediate additional output
before returning. A byte transport cannot predict output that a child has not
yet produced: delayed unsolicited bytes are detected on the next exchange.
`OfficialTelemetrySink` additionally validates JSON schemas, request identity,
receipts and cumulative counters. Child diagnostics must use stderr.

Stderr bytes are discarded without being copied into replies, exceptions or
logs. A cumulative 64 KiB default quota detects a diagnostic flood. Detached
statistics expose only completed exchange count and drained byte counts. Error
details use stable operation names and contain no argv, environment, path,
stdout, stderr or native error-message text.

The maximum launch byte quota includes argv and the child environment. Windows
also enforces its UTF-16 command-line bound. Request/reply/stderr/launch quotas
must be positive and no larger than 1 GiB. They are ceilings, not preallocations.

## Cancellation, deadlines and restart

An operation uses the earlier of its supplied absolute steady-clock deadline and
`operation_timeout` (10 seconds by default). Startup similarly uses
`startup_timeout`. Stop/deadline checks cover pipe writes, reply collection,
stderr drain and the extra-output quiet period. Native nonblocking/overlapped
pipe waits are at most 5 ms, so cancellation does not depend on the child
cooperating or closing a pipe. Native process-creation calls themselves are OS
calls; late return is detected before a launch can be reported successful.

Any failed exchange, including invalid input or a pre-request cancellation,
poisons and terminates the transport. Further exchanges throw `NotReady`.
Construct a new transport to restart. Never silently restart within `transact`:
the failed child may already have exported a prefix. Retain and replay the
original immutable durable observations using the sink's at-least-once contract.

`terminate()` and destruction release ownership without throwing. They do not
send a protocol close/flush. Call the observer's explicit `close` first when its
final counters and export acknowledgement are needed, then release the channel.
Process termination/reaping may extend an already reached operation deadline by
the bounded cleanup grace (2 seconds by default). Native cleanup errors never
replace the original exchange failure or leak diagnostics.

Timeouts must be positive: operation/startup at most 10 minutes, cleanup at most
10 seconds, and the reply quiet period at most 100 ms. Increase these values
explicitly for a measured workload; no wait silently becomes infinite.

## Platform containment

Windows uses Unicode `CreateProcessW`, Microsoft CRT argv quoting, an explicit
standard-handle inheritance list and overlapped local named pipes. The process
starts suspended, joins a kill-on-close Job Object, then resumes. Neither
breakaway nor silent-breakaway is enabled. If job assignment is unavailable,
launch fails and the suspended child is terminated; there is no uncontained
fallback. Nested child processes remain in the job. Closing the owning job also
terminates descendants that retain inherited pipe handles.

POSIX uses a distinct process group, fixed absolute `execve`, parent-only
nonblocking descriptors, `poll`, and an exec-status pipe. Launch success requires
the close-on-exec status pipe to close without an error record. The fork child
uses only prepared data and async-signal-safe system calls. Unrelated parent
descriptors are closed or marked close-on-exec. SIGPIPE is masked only in the
writing thread; the library does not change process-wide signal handlers.

Cleanup sends SIGKILL to the entire original group and reaps its direct child.
Exit observation keeps the leader's PID reserved until group cleanup. Descendant
zombies are reaped by their actual parent or system reaper; the transport does
not change the application's global subreaper or SIGCHLD policy. The caller must
not reap this transport's child from an unrelated global wait handler. Launch
rejects an existing `SIGCHLD=SIG_IGN` or `SA_NOCLDWAIT` policy because automatic
reaping would invalidate process ownership before group cleanup. POSIX
process groups supervise cooperative official sidecars; they are not a sandbox
for hostile programs that deliberately escape with `setsid` or `setpgid`.

## Validation and build integration

`tests/cli/test_sidecar_transport.cpp` starts the actual Python fixture under
`tests/cli/sidecar_fixture.py`. It covers persistent echo, Unicode paths and argv
quoting, child-only environment/cwd, explicit/default deadline, cancellation,
blocked stdin, EOF, partial/extra/late frames, invalid UTF-8/NUL/CR, stdout/stderr
floods, crash, poison/restart, failed exec, inherited-handle exclusion, descendant
termination and Windows Job breakaway refusal.

The CMake test target should compile the product
`tools/cli/sidecar_process.cpp`, link the experiment/foundation implementation,
GTest, bounded JSON implementation and `Threads::Threads`, and define:

```cmake
JYPPX_SIDECAR_TEST_PYTHON="${Python3_EXECUTABLE}"
JYPPX_SIDECAR_TEST_FIXTURE="${CMAKE_SOURCE_DIR}/tests/cli/sidecar_fixture.py"
```

Use CMake's normalized absolute paths for those UTF-8 definitions. Explicit
environment variables of the same names are a fallback for direct invocation.
Absent Python or fixture prerequisites fail the test; they do not skip it.
Windows requires the normal Kernel32 process/job/named-pipe APIs. Linux uses
pthread signal-mask APIs; no global signal disposition is modified.

This module and its tests were developed without per-edit compilation. The
chapter's unified Windows and remote Ubuntu/CUDA validation must register and
execute these targets, alongside the official TensorBoard/Collector consumer
tests, before the telemetry chapter is accepted.
