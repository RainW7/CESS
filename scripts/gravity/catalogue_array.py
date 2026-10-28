#!/usr/bin/env python3
"""Prepare and run one complete catalogue per Gravity/Torque array element.

This is a submission helper, not a new scientific model or an MPI launcher.
Preparation reads catalogue metadata, not the multi-terabyte spectral payload.
Only the user invokes qsub. Each compute task delegates verification and resume
to the unmodified CESS CLI with its own manifest.
"""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


GROUPS = ("MzLS", "DECaLS")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
RUNTIME_PATHS = {"input_path", "output_path", "el_input_path", "el_output_path"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    """Atomic replacement for operational status; never used for science data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_identity():
    packages = {}
    for name in ("numpy", "scipy", "astropy", "h5py", "specutils", "tqdm", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "packages": packages}


def source_hashes(repo):
    return {str(path): sha256(path) for path in sorted((repo / "src/cess").rglob("*.py"))}


def install_source(repo):
    require(sys.version_info[:2] == (3, 12), "Use the CESS Python 3.12 interpreter")
    require((repo / "src/cess").is_dir(), "--repo must contain src/cess")
    sys.path.insert(0, str(repo / "src"))
    import cess
    from cess.contracts import REPAIR_REVISION
    require(Path(cess.__file__).resolve().parent == repo / "src/cess", "Wrong imported CESS source")
    require(cess.__version__ == "0.9.0", "This helper requires CESS 0.9.0")
    require(REPAIR_REVISION == "20261027-D5", "Unexpected CESS scientific revision")


def walltime(value):
    require(re.fullmatch(r"\d{1,2}:[0-5]\d:[0-5]\d", value) is not None,
            "Walltime must use HH:MM:SS")
    hours, minutes, seconds = map(int, value.split(":"))
    require(0 < hours * 3600 + minutes * 60 + seconds <= 48 * 3600,
            "normal queue walltime must be positive and at most 48:00:00")
    return value


def check_catalogue_metadata(path, rows):
    """Check source alignment by shape, allowing invalid numerical source rows."""
    import h5py
    with h5py.File(path, "r") as handle:
        require(handle["ID"].shape == (rows,), "ID must be one-dimensional: " + str(path))
        require("parameters" in handle and handle["parameters"].ndim == 2 and
                handle["parameters"].shape[0] == rows, "Unaligned parameters: " + str(path))
        require("parameters_name" in handle and handle["parameters_name"].shape ==
                (handle["parameters"].shape[1],), "Unaligned parameter names: " + str(path))
        best = handle["best_fit"]
        require(best["wavelength_rest"].ndim == 1, "Wavelength grid must be one-dimensional: " + str(path))
        require(best["z"].shape == (rows,), "Unaligned redshifts: " + str(path))
        spectrum = next((name for name in best if name.startswith("spec_")), None)
        require(spectrum is not None and best[spectrum].shape == (rows, len(best["wavelength_rest"])),
                "Missing or unaligned spec_* dataset: " + str(path))


def pbs_script(plan, group, plan_digest):
    settings = plan["groups"][group]
    run_dir = Path(plan["run_dir"])
    command = [plan["python"], "-s", "-B", plan["runner"], "run", "--plan",
               str(run_dir / "plan.json"), "--plan-sha256", plan_digest, "--group", group]
    return "\n".join([
        "#!/bin/bash", "#PBS -N cess_" + group.lower(), "#PBS -S /bin/bash", "#PBS -q normal",
        "#PBS -l nodes=1:ppn=1", "#PBS -l mem={}gb".format(settings["memory_gb"]),
        "#PBS -l walltime=" + settings["walltime"], "#PBS -j oe", "#PBS -r n",
        "#PBS -t 1-{}%{}".format(len(settings["jobs"]), settings["slots"]),
        "#PBS -o " + shlex.quote(str(run_dir / "pbs_logs") + "/"),
        "", "set -Eeuo pipefail", "umask 027",
        "export " + " ".join(key + "=1" for key in THREAD_KEYS),
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1",
        "export PYTHONPATH=" + shlex.quote(str(Path(plan["repo"]) / "src")),
        "cd -- " + shlex.quote(plan["repo"]),
        "exec " + shlex.join(command), ""])


def prepare(args):
    repo = args.repo.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else run_dir / "products"
    require(not (run_dir.exists() or run_dir.is_symlink()), "Run directory already exists; use a new --run-dir")
    require(not (output.exists() or output.is_symlink()), "Output path already exists; use a new output directory")
    for path in (repo, run_dir, output, Path(sys.executable)):
        require(not any(character in str(path) for character in "\r\n"), "Newlines in paths are unsupported")
    install_source(repo)
    from cess.cli.main import build_jobs, discover_inputs
    from cess.config.loader import DEFAULT_CONFIG_PATH, load_config
    inputs = discover_inputs(args.input)
    # Capture an explicit, fully resolved configuration. No environment-only
    # configuration selection and no paths relative to the submission cwd.
    config = {key: value for key, value in load_config(args.config or DEFAULT_CONFIG_PATH).items()
              if not key.startswith("_")}
    config.update(input_path=str(inputs[0].parent), output_path=str(output),
                  el_input_path=str(inputs[0].parent), el_output_path=str(output))
    options = dict(apply_morphology_effect=True, apply_wavelength_calibration_error=True,
                   apply_photometric=True, apply_el_detect=True)
    jobs = build_jobs(inputs, output, config, options, mode="standard", seed=args.seed,
                      output_layout="structured")
    groups = {}
    for group in GROUPS:
        prefix = group.lower()
        memory = getattr(args, prefix + "_memory_gb")
        slots = getattr(args, prefix + "_slots")
        require(1 <= memory <= 360, "normal queue memory must be between 1 and 360 GB")
        require(slots > 0, "Array slot limit must be positive")
        groups[group] = dict(memory_gb=memory, slots=slots, budget_gib=0.85 * memory,
                             walltime=walltime(getattr(args, prefix + "_walltime")), jobs=[])
    for path, job in zip(inputs, jobs):
        check_catalogue_metadata(path, job["selected_rows"])
        candidates = [group for group in GROUPS if group.lower() in path.name.lower()]
        require(len(candidates) == 1, "Cannot unambiguously classify input survey: " + str(path))
        group = candidates[0]
        stat = path.stat()
        groups[group]["jobs"].append(dict(input=str(path), rows=job["selected_rows"],
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, estimated_peak_gib=job["estimated_peak_gib"]))
    for group, settings in groups.items():
        expected = getattr(args, "expected_" + group.lower())
        require(expected >= 0, "Expected input counts must be nonnegative")
        require(len(settings["jobs"]) == expected,
                "{} input count: expected {}, found {}".format(group, expected, len(settings["jobs"])))
        settings["jobs"].sort(key=lambda job: (-job["estimated_peak_gib"], job["input"]))
        for index, job in enumerate(settings["jobs"], 1):
            job["index"] = index
            require(job["rows"] > 0, "Empty input catalogue: " + job["input"])
            require(job["estimated_peak_gib"] <= settings["budget_gib"],
                    "{} requires estimated {:.2f} GiB, above {:.2f} GiB budget; "
                    "increase --{}-memory-gb before preparing".format(
                        job["input"], job["estimated_peak_gib"], settings["budget_gib"], group.lower()))
    resources = {}
    for name, value in config.items():
        if name not in RUNTIME_PATHS and name.endswith(("_path", "_pkl")):
            path = Path(value)
            require(path.is_file(), "Missing CESS resource: " + str(path))
            resources[str(path)] = sha256(path)
    # All inventory/resource checks precede any creation of the campaign.
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("pbs_logs", "logs", "status", "manifests"):
        (run_dir / directory).mkdir()
    write_json(run_dir / "config.json", config)
    runner = run_dir / "catalogue_array.py"
    shutil.copyfile(Path(__file__).resolve(), runner)
    plan = dict(format_version=1, created_at=utc_now(), repo=str(repo), run_dir=str(run_dir),
        output=str(output), python=sys.executable, runner=str(runner), runner_sha256=sha256(runner),
        config=str(run_dir / "config.json"), config_sha256=sha256(run_dir / "config.json"),
        source_hashes=source_hashes(repo), resource_hashes=resources, runtime=runtime_identity(),
        version="0.9.0", revision="20261027-D5", seed=args.seed, groups=groups,
        mode="standard", output_layout="structured", all_effects=True, workers_per_task=1)
    write_json(run_dir / "plan.json", plan)
    digest = sha256(run_dir / "plan.json")
    (run_dir / "plan.sha256").write_text(digest + "  plan.json\n")
    for group, settings in groups.items():
        if settings["jobs"]:
            (run_dir / (group + ".pbs")).write_text(pbs_script(plan, group, digest))
    print(json.dumps({"plan": str(run_dir / "plan.json"),
        "files": {group: len(settings["jobs"]) for group, settings in groups.items()},
        "input_bytes": sum(job["bytes"] for settings in groups.values() for job in settings["jobs"]),
        "source_rows": sum(job["rows"] for settings in groups.values() for job in settings["jobs"]),
        "max_simultaneous_cpus": sum(settings["slots"] for settings in groups.values() if settings["jobs"]),
        "pilot_commands": [shlex.join(["qsub", "-t", "1-1%1", str(run_dir / (group + ".pbs"))])
                           for group in GROUPS if groups[group]["jobs"]],
        "full_commands": [shlex.join(["qsub", str(run_dir / (group + ".pbs"))])
                          for group in GROUPS if groups[group]["jobs"]]}, indent=2))


def load_plan(path, expected_digest=None):
    path = path.expanduser().resolve()
    expected = expected_digest or (path.parent / "plan.sha256").read_text().split()[0]
    require(sha256(path) == expected, "Plan checksum mismatch; do not edit a prepared campaign")
    plan = json.loads(path.read_text())
    require(plan["format_version"] == 1, "Unsupported array plan version")
    require(Path(plan["run_dir"]) == path.parent, "Do not move a prepared campaign")
    return plan


def check_identity(plan, job):
    require(sha256(plan["runner"]) == plan["runner_sha256"], "Frozen runner checksum mismatch")
    require(Path(sys.executable).resolve() == Path(plan["python"]).resolve(), "Python interpreter path changed")
    require(runtime_identity() == plan["runtime"], "Python or dependency versions changed")
    require(source_hashes(Path(plan["repo"])) == plan["source_hashes"], "CESS source changed after preparation")
    require(sha256(plan["config"]) == plan["config_sha256"], "Frozen config checksum mismatch")
    for path, digest in plan["resource_hashes"].items():
        require(sha256(path) == digest, "CESS resource checksum changed: " + path)
    stat = Path(job["input"]).stat()
    require((stat.st_size, stat.st_mtime_ns) == (job["bytes"], job["mtime_ns"]),
            "Input file size/mtime changed after preparation: " + job["input"])


def allocation():
    require(os.environ.get("PBS_ENVIRONMENT") == "PBS_BATCH" and os.environ.get("PBS_JOBID"),
            "Actual runs require a PBS batch job; use --dry-run to preview on a login node")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    require(nodefile and Path(nodefile).is_file(), "Missing readable PBS_NODEFILE")
    nodes = Path(nodefile).read_text().split()
    require(len(nodes) == 1, "Request exactly one host and one CPU per array element")
    for key in ("PBS_NP", "PBS_NUM_NODES", "num_nodes"):
        if key in os.environ:
            require(os.environ[key] == "1", key + " must equal 1 for this one-file, one-CPU payload")
    return {key: os.environ.get(key) for key in
            ("PBS_JOBID", "PBS_ARRAYID", "PBS_NODEFILE", "PBS_NP", "PBS_NUM_NODES")}


@contextmanager
def task_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Array task is already running; do not remove its lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(args):
    plan = load_plan(args.plan, args.plan_sha256)
    group = plan["groups"][args.group]
    raw_index = os.environ.get("PBS_ARRAYID")
    if not args.dry_run:
        pbs = allocation()
        require(raw_index is not None, "Torque must provide PBS_ARRAYID")
        require(args.index is None or str(args.index) == raw_index, "--index differs from PBS_ARRAYID")
    else:
        pbs = {}
    index = args.index if args.index is not None else int(raw_index or "0")
    require(1 <= index <= len(group["jobs"]), "Array index is outside the prepared input list")
    job = group["jobs"][index - 1]
    check_identity(plan, job)
    run_dir = Path(plan["run_dir"])
    key = "{:04d}".format(index)
    manifest = run_dir / "manifests" / args.group / (key + ".json")
    command = [plan["python"], "-s", "-B", "-m", "cess", "simulate",
        "--config", plan["config"], "-i", job["input"],
        "-o", str(Path(plan["output"]) / args.group), "--all-effects",
        "--mode", "standard", "--output-layout", "structured", "--workers", "1",
        "--threads-per-worker", "1", "--seed", str(plan["seed"]),
        "--memory-budget-gib", str(group["budget_gib"])]
    if args.dry_run:
        # A new dry-run must not pass --manifest: the CLI would create one.
        if manifest.exists():
            command += ["--manifest", str(manifest), "--resume"]
        return subprocess.run(command + ["--dry-run"], cwd=plan["repo"],
                              env=child_environment(plan)).returncode
    state_path = run_dir / "status" / args.group / (key + ".json")
    with task_lock(state_path.with_suffix(".lock")):
        # Decide resume while holding the wrapper lock: an earlier invocation
        # may have created its manifest just before releasing this same lock.
        command += ["--manifest", str(manifest)]
        if manifest.exists():
            command += ["--resume"]
        log_dir = run_dir / "logs" / args.group
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = "{}_{}_{}".format(key, time.time_ns(), os.getpid())
        log_path = log_dir / (attempt + ".log")
        attempt_path = log_dir / (attempt + ".json")
        state = dict(state="running", group=args.group, index=index, input=job["input"],
                     rows=job["rows"], host=socket.gethostname(), started_at=utc_now(),
                     pbs=pbs, command=command, log=str(log_path), manifest=str(manifest),
                     estimated_peak_gib=job["estimated_peak_gib"], memory_requested_gb=group["memory_gb"])
        write_json(state_path, state)
        write_json(attempt_path, state)
        print("CESS task log: " + str(log_path), flush=True)
        child = None
        interrupted = []

        def forward(signum, frame):
            interrupted.append(signum)
            if child is not None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

        previous = {number: signal.signal(number, forward) for number in (signal.SIGINT, signal.SIGTERM)}
        start = time.monotonic()
        try:
            with log_path.open("x") as log:
                log.write(json.dumps(state, sort_keys=True) + "\n")
                log.flush()
                require(not interrupted, "Interrupted before starting the CESS process")
                # Keep the PBS session so Torque can clean up the whole job
                # even if this controller is killed. A separate process group
                # still permits forwarding cancellation to executor workers.
                child = subprocess.Popen(command, cwd=plan["repo"], env=child_environment(plan),
                    stdout=log, stderr=subprocess.STDOUT, process_group=0)
                if interrupted:
                    forward(interrupted[-1], None)
                result = child.wait()
                if interrupted:
                    # Include executor workers even if their CLI parent exited
                    # first; cancellation must not leave orphan simulations.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    result = 128 + interrupted[-1]
                state.update(state="interrupted" if interrupted else "complete" if result == 0 else "failed",
                             returncode=result)
        except BaseException as error:
            state.update(state="failed", returncode=1, error=type(error).__name__ + ": " + str(error))
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            state.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - start,
                         max_child_rss_mib=maximum / (1024 ** 2 if sys.platform == "darwin" else 1024),
                         rss_definition="maximum child resident set; not summed concurrent job memory")
            write_json(attempt_path, state)
            write_json(state_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
        return state["returncode"] if state["returncode"] >= 0 else 128 - state["returncode"]


def child_environment(plan):
    environment = os.environ.copy()
    environment.update({key: "1" for key in THREAD_KEYS})
    environment.update(PYTHONPATH=str(Path(plan["repo"]) / "src"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    return environment


def index_ranges(indices):
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][-1] + 1:
            ranges[-1].append(index)
        else:
            ranges.append([index])
    return ",".join(str(item[0]) if len(item) == 1 else "{}-{}".format(item[0], item[-1])
                    for item in ranges)


def status(args):
    plan = load_plan(args.plan)
    result = dict(groups={}, retry_commands={}, retry_requires_stopped_jobs=True,
                  note="Status files are operational records; CESS resume revalidates products and hashes.")
    for group, settings in plan["groups"].items():
        counts = dict(total=len(settings["jobs"]), complete=0, failed=0, interrupted=0, running=0, pending=0)
        retry = []
        for job in settings["jobs"]:
            path = Path(plan["run_dir"]) / "status" / group / ("{:04d}.json".format(job["index"]))
            value = json.loads(path.read_text()).get("state", "pending") if path.exists() else "pending"
            require(value in counts and value != "total", "Invalid status file: " + str(path))
            counts[value] += 1
            if value != "complete":
                retry.append(job["index"])
        result["groups"][group] = counts
        if retry:
            result["retry_commands"][group] = shlex.join(["qsub", "-t",
                index_ranges(retry) + "%" + str(settings["slots"]), str(Path(plan["run_dir"]) / (group + ".pbs"))])
    print(json.dumps(result, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    prepare_parser = commands.add_parser("prepare", help="Inventory inputs and generate two Torque array scripts; never submit")
    prepare_parser.add_argument("-i", "--input", nargs="+", required=True)
    prepare_parser.add_argument("--run-dir", type=Path, required=True, help="New absolute campaign directory")
    prepare_parser.add_argument("-o", "--output", type=Path, help="New output directory (default RUN/products)")
    prepare_parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    prepare_parser.add_argument("--config", type=Path, help="Instrument JSON; default is bundled configuration")
    prepare_parser.add_argument("--seed", type=int, default=1214)
    for group, memory, duration in (("mzls", 16, "12:00:00"), ("decals", 48, "24:00:00")):
        prepare_parser.add_argument("--expected-" + group, type=int, default=1385)
        prepare_parser.add_argument("--" + group + "-slots", type=int, default=32)
        prepare_parser.add_argument("--" + group + "-memory-gb", type=int, default=memory)
        prepare_parser.add_argument("--" + group + "-walltime", default=duration)
    run_parser = commands.add_parser("run", help="Execute exactly one array element; normally invoked by PBS")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--plan-sha256")
    run_parser.add_argument("--group", choices=GROUPS, required=True)
    run_parser.add_argument("--index", type=int, help="Preview index; actual jobs use Torque PBS_ARRAYID")
    run_parser.add_argument("--dry-run", action="store_true")
    status_parser = commands.add_parser("status", help="Report local task records and print retry commands; never submit")
    status_parser.add_argument("--plan", type=Path, required=True)
    return result


def main(argv=None):
    # Set these before any scientific-library import in prepare or a dry-run.
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    args = parser().parse_args(argv)
    try:
        if args.action == "prepare":
            require(args.seed >= 0, "Seed must be nonnegative")
            prepare(args)
        elif args.action == "run":
            return run(args)
        else:
            status(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print("CESS Gravity array: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
