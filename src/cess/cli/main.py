"""Explicit, bounded command line entry point for local CESS simulations.

Example: python /path/to/cess_cli.py simulate --config parameters.json
         --input 'input/*.hdf5' --output results --workers 1 --seed 1214 --dry-run
No configuration or scientific resource is loaded merely by importing this file.

Manifests are per-run checkpoints: each job rewrites the complete JSON atomically.
For catalogue production, use one manifest per input file rather than one manifest
for all inputs/chunks; this format is not an efficient 138000-job event journal.
Resume only after workers from the previous invocation have stopped.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import glob
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import h5py


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def discover_inputs(specifications, default_directory=None):
    """Resolve directories/globs/files; each explicit specification must match."""
    specifications = specifications or ([default_directory] if default_directory else [])
    found = []
    for specification in specifications:
        path = Path(specification).expanduser()
        if path.is_dir():
            matches = sorted(list(path.glob("*.hdf5")) + list(path.glob("*.h5")))
        elif path.is_file():
            matches = [path]
        else:
            matches = [Path(item) for item in sorted(glob.glob(str(path)))
                       if Path(item).is_file()]
        if not matches:
            raise ValueError("No input HDF5 files match {!r}".format(specification))
        found.extend(path.resolve() for path in matches)
    unique = sorted(set(found))
    if not unique:
        raise ValueError("No input files selected")
    stems = [path.stem for path in unique]
    if len(set(stems)) != len(stems):
        raise ValueError("Distinct inputs with the same output stem need separate output runs")
    return unique


def estimate_peak_gib(config, rows, spectral_length, parameter_columns):
    """Approximate working memory for scheduling, not a measured upper bound."""
    pixels = sum(round((int((config[b + "_end_wl"] - config[b + "_start_wl"]) /
                               config[b + "_dlambda"]) + 1) *
                       config[b + "_ree80"] / config["arcsecperpix"])
                 for b in ("gu", "gv", "gi"))
    # Include input copies, parameters, many spectral products and headroom.
    per_row = 8 * (3 * spectral_length + parameter_columns + 20 * pixels)
    return (1.5 * rows * per_row + 160 * 1024 ** 2) / 1024 ** 3


def build_jobs(inputs, output, config, options, mode="standard", seed=0,
               source_start=0, source_stop=None, chunk_size=None, output_layout='legacy'):
    # A saved job plan is explicit and must never inherit a worker's last task.
    from cess.pipeline.support import normalise_options
    from cess.pipeline.common import validate_output_layout
    from cess.io.products import open_product
    validate_output_layout(output_layout)
    options = normalise_options(options)
    if source_stop is not None and source_stop < source_start:
        raise ValueError("source-stop must be at least source-start")
    jobs = []
    for path in inputs:
        source = open_product(path) if mode == 'el-detect' else h5py.File(str(path), 'r')
        with source as product:
            handle = product.legacy_view() if mode == 'el-detect' else product
            if "ID" not in handle:
                raise ValueError("Input is missing ID: {}".format(path))
            count = len(handle["ID"])
            if source_start > count or (source_stop is not None and source_stop > count):
                raise ValueError("Requested source interval exceeds {} IDs in {}".format(count, path))
            end = count if source_stop is None else source_stop
            spectral_length, columns = 0, 0
            if "best_fit/wavelength_rest" in handle:
                spectral_length = len(handle["best_fit/wavelength_rest"])
            if "parameters" in handle and handle["parameters"].ndim == 2:
                columns = handle["parameters"].shape[1]
            if mode != "el-detect" and (not spectral_length or "best_fit/z" not in handle):
                raise ValueError("Expected a BayeSED best_fit SED input: {}".format(path))
        size = chunk_size or max(1, end - source_start)
        intervals = [(begin, min(begin + size, end)) for begin in range(source_start, end, size)]
        if end == source_start:
            intervals = [(source_start, end)]
        for begin, end_row in intervals:
            jobs.append({"input_path": str(path.parent) + "/", "filename": path.name,
                         "output_dir": str(output), "config": config, "options": options,
                         "mode": mode, "output_layout": output_layout, "seed": seed, "source_start": begin,
                         "source_stop": end_row, "selected_rows": end_row - begin,
                         "estimated_peak_gib": estimate_peak_gib(
                             config, end_row - begin, spectral_length, columns)})
    return jobs


def _execute_job(job):
    started = time.monotonic()
    progress = dict(worker_pid=os.getpid(), input=job['filename'],
                    source_start=job['source_start'], source_stop=job['source_stop'])
    print(json.dumps(dict(progress, event='cess_job_started', monotonic_s=started)), flush=True)
    module_name = {"standard": "cess.pipeline.standard", "paired-wce": "cess.pipeline.paired",
                   "el-detect": "cess.pipeline.emission"}[job["mode"]]
    driver = importlib.import_module(module_name)
    result = driver.simulate_file(job["input_path"], job["filename"],
                         output_dir=job["output_dir"], config=job["config"],
                         options=job["options"], seed=job["seed"],
                         source_start=job["source_start"], source_stop=job["source_stop"],
                         output_layout=job.get('output_layout', 'legacy'))
    finished = time.monotonic()
    print(json.dumps(dict(progress, event='cess_job_finished', monotonic_s=finished,
                          elapsed_seconds=finished-started)), flush=True)
    return str(result)


def parser(default_layout='structured'):
    root = argparse.ArgumentParser(description="CESS reproducible, bounded batch runner")
    from cess._version import __version__
    root.add_argument('--version', action='version', version='CESS ' + __version__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("simulate", "el-detect"):
        item = commands.add_parser(command)
        item.add_argument("--config", type=Path, help="JSON config; resources relative to this file")
        item.add_argument("-i", "--input", nargs="+", help="HDF5 files, directories or quoted globs")
        item.add_argument("-o", "--output", type=Path, help="Output directory; existing products are never overwritten")
        item.add_argument('--output-layout', choices=('legacy', 'structured'), default=default_layout,
                          help='HDF5 layout: structured schema 3.0 or compatible legacy schema 2.5')
        item.add_argument("-j", "--workers", type=_positive, default=1,
                          help="Maximum simultaneous file/chunk processes (default: 1)")
        item.add_argument("--threads-per-worker", type=_positive, default=1,
                          help="Native numeric-library threads requested before each worker starts (default: 1)")
        item.add_argument("--seed", type=_nonnegative, default=0)
        item.add_argument("--source-start", type=_nonnegative, default=0)
        item.add_argument("--source-stop", type=_nonnegative)
        item.add_argument("--chunk-size", type=_positive, help="Maximum original source rows per output product")
        item.add_argument("--memory-budget-gib", type=float,
                          help="Limit concurrency using estimated working memory (not an OS hard limit)")
        item.add_argument("--dry-run", action="store_true", help="Print the plan without running simulations")
        item.add_argument("--manifest", type=Path,
                          help="Per-run JSON checkpoint, fully rewritten per job; use one per input file in production")
        item.add_argument("--resume", action="store_true",
                          help="Resume the identical saved --manifest after previous workers have stopped")
        if command == "simulate":
            item.add_argument("--mode", choices=("standard", "paired-wce"), default="standard")
            item.add_argument("--all-effects", action="store_true",
                              help="Enable morphology, wavelength error, photometry and emission lines together")
            item.add_argument("--morphology", action="store_true")
            item.add_argument("--wavelength-error", action="store_true")
            item.add_argument("--photometry", action="store_true")
            item.add_argument("--emission-lines", action="store_true")
    merge = commands.add_parser("merge", help="Combine simulation chunks into one self-contained HDF5 file")
    merge.add_argument("-i", "--input", nargs="+", required=True,
                       help="Chunks from one simulation: files, directory or quoted glob")
    merge.add_argument("-o", "--output", type=Path, required=True,
                       help="New output HDF5 file; existing paths are never overwritten")
    merge.add_argument("--input-source", type=Path,
                       help="Original BayeSED HDF5 to verify complete input coverage and IDs")
    merge.add_argument("--copy-rows", type=_positive, default=128,
                       help="Maximum rows copied at once (default: 128); does not split the output")
    merge.add_argument("--dry-run", action="store_true",
                       help="Validate chunks and print the merge plan without creating output")
    return root


_JOB_IDENTITY_KEYS = ('input_path', 'filename', 'output_dir', 'mode', 'seed',
                      'source_start', 'source_stop', 'selected_rows', 'output_layout')


def _result_key(result):
    if (not isinstance(result.get('input'), str) or
            any(type(result.get(name)) is not int for name in ('source_start', 'source_stop')) or
            result['source_start'] < 0 or result['source_stop'] < result['source_start']):
        raise ValueError('Saved manifest contains an invalid job identity')
    return result['input'], result['source_start'], result['source_stop']


def _pending_result(job):
    return dict(input=job['filename'], source_start=job['source_start'],
                source_stop=job['source_stop'], status='pending')


def _resume_plan(path, summary, jobs, verify_product, expected_product):
    """Validate the complete plan and all existing products before scheduling."""
    with path.open() as stream:
        saved = json.load(stream)
    if (not isinstance(saved, dict) or type(saved.get('manifest_version')) is not int or
            saved['manifest_version'] != 1 or
            not isinstance(saved.get('snapshot'), dict)):
        raise ValueError('Cannot resume legacy/unversioned manifest without a snapshot')
    if saved['snapshot'] != summary['snapshot']:
        raise ValueError('Resume snapshot mismatch: configuration, options, seed, mode, '
                         'input, code or resources changed')
    planned = [{key: job[key] for key in _JOB_IDENTITY_KEYS} for job in jobs]
    try:
        previous = [{key: job[key] for key in _JOB_IDENTITY_KEYS} for job in saved['jobs']]
    except (KeyError, TypeError):
        raise ValueError('Saved manifest is missing a complete job plan')
    if previous != planned:
        raise ValueError('Resume job plan mismatch: source intervals or output directory changed')
    old_results = saved.get('results')
    if not isinstance(old_results, list):
        raise ValueError('Saved manifest is missing job states')
    states = {}
    for result in old_results:
        if not isinstance(result, dict) or result.get('status') not in ('pending', 'complete', 'failed'):
            raise ValueError('Saved manifest contains an invalid job state')
        key = _result_key(result)
        if key in states:
            raise ValueError('Saved manifest contains duplicate job states')
        states[key] = result
    if set(states) != {_result_key(_pending_result(job)) for job in jobs}:
        raise ValueError('Saved manifest job states do not match its job plan')
    results, pending = [], []
    for job in jobs:
        state = states[_result_key(_pending_result(job))]
        if state['status'] == 'complete' and not state.get('sha256'):
            raise ValueError('Completed manifest job is missing its output checksum')
        if 'output' in state:
            if not isinstance(state['output'], str) or Path(state['output']).resolve() != expected_product(job).resolve():
                raise ValueError('Saved result output does not match its planned destination')
        verified = verify_product(job, summary['snapshot'], expected_sha256=state.get('sha256'))
        result = _pending_result(job)
        if verified is None:
            pending.append(job)
            if state['status'] == 'complete':
                result['resume_reason'] = 'previously complete product is missing'
        else:
            result.update(verified, status='complete')
            if state['status'] != 'complete':
                result['recovered_orphan'] = True
        results.append(result)
    summary['results'] = results
    summary['resumed'] = True
    summary['pending_jobs'] = len(pending)
    return pending


def _run_jobs(jobs, summary, workers, persist=None, verify_product=None, threads_per_worker=1):
    """Keep work bounded and persist every individual terminal job state."""
    failed = False
    indices = {_result_key(result): index for index, result in enumerate(summary['results'])}
    context = multiprocessing.get_context('spawn')
    from cess.cli.parallel import worker_thread_environment
    with worker_thread_environment(threads_per_worker), \
            ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        remaining = iter(jobs)
        futures = {}

        def refill():
            while len(futures) < 2 * workers:
                job = next(remaining, None)
                if job is None:
                    break
                futures[pool.submit(_execute_job, job)] = job

        refill()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                result = _pending_result(job)
                try:
                    output = future.result()
                    if verify_product is None:
                        result.update(status='complete', output=output)
                    else:
                        verified = verify_product(job, summary['snapshot'])
                        if verified is None or Path(output).resolve() != Path(verified['output']).resolve():
                            raise ValueError('Worker did not publish its expected verified product')
                        result.update(verified, status='complete')
                except Exception as exc:
                    failed = True
                    result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                key = _result_key(result)
                if key in indices:
                    summary['results'][indices[key]] = result
                else:
                    indices[key] = len(summary['results'])
                    summary['results'].append(result)
                if persist is not None:
                    summary['pending_jobs'] = sum(item['status'] == 'pending' for item in summary['results'])
                    persist()
                print(json.dumps(result), flush=True)
            refill()
    return 1 if failed else 0


def main(argv=None, default_layout='structured'):
    arguments = parser(default_layout).parse_args(argv)
    from cess.config.loader import load_config
    try:
        if arguments.command == "merge":
            from cess.io.merge import merge_products
            sources = discover_inputs(arguments.input)
            if arguments.input_source is not None:
                plan = merge_products(sources, arguments.output,
                                      expected_input=arguments.input_source,
                                      chunk_rows=arguments.copy_rows, dry_run=True)
                if plan['covers_entire_input'] is not True:
                    raise ValueError('--input-source requires chunks covering every row of the input')
                if arguments.dry_run:
                    print(json.dumps(plan, indent=2), flush=True)
                    return 0
            result = merge_products(sources, arguments.output,
                                    expected_input=arguments.input_source,
                                    chunk_rows=arguments.copy_rows,
                                    dry_run=arguments.dry_run)
            print(json.dumps(result if arguments.dry_run else
                             {"status": "complete", "output": str(result)}, indent=2), flush=True)
            return 0
        if arguments.resume and arguments.manifest is None:
            raise ValueError('--resume requires --manifest PATH')
        config = load_config(arguments.config)
        is_el = arguments.command == "el-detect"
        mode = "el-detect" if is_el else arguments.mode
        all_effects = getattr(arguments, 'all_effects', False)
        options = {"morph": all_effects or getattr(arguments, "morphology", False),
                   "wave_cal": all_effects or getattr(arguments, "wavelength_error", False),
                   "photo": all_effects or getattr(arguments, "photometry", False),
                   "el_detect": is_el or all_effects or getattr(arguments, "emission_lines", False)}
        inputs = discover_inputs(arguments.input,
                                 config["el_input_path" if is_el else "input_path"])
        output = (arguments.output or Path(config["el_output_path" if is_el else "output_path"])).resolve()
        jobs = build_jobs(inputs, output, config, options, mode, arguments.seed,
                          arguments.source_start, arguments.source_stop, arguments.chunk_size,
                          output_layout=arguments.output_layout)
        workers = min(arguments.workers, len(jobs))
        if arguments.memory_budget_gib is not None:
            if not math.isfinite(arguments.memory_budget_gib) or arguments.memory_budget_gib <= 0:
                raise ValueError("memory-budget-gib must be positive and finite")
            affordable = int(arguments.memory_budget_gib / max(job["estimated_peak_gib"] for job in jobs))
            if affordable < 1:
                raise ValueError("A task exceeds the estimated memory budget; reduce --chunk-size")
            workers = min(workers, affordable)
        if arguments.manifest and arguments.manifest.exists() and not arguments.resume:
            raise FileExistsError("Manifest already exists: {}".format(arguments.manifest))
        summary = {"workers": workers, "seed": arguments.seed, "mode": mode,
                   'requested_workers': arguments.workers,
                   'threads_per_worker': arguments.threads_per_worker,
                   'input_files': len(inputs), 'task_count': len(jobs),
                   'output_layout': arguments.output_layout,
                   "configuration": config, "options": options,
                   "memory_estimate_is_hard_limit": False,
                   "jobs": [{key: value for key, value in job.items()
                             if key not in ('config', 'options')} for job in jobs],
                   "results": []}
        if not arguments.dry_run:
            print('CESS: {} input file(s), {} task(s), {} worker(s) (requested {}), '
                  '{} native thread(s)/worker; {}'.format(
                      len(inputs), len(jobs), workers, arguments.workers,
                      arguments.threads_per_worker,
                      'whole-file tasks' if arguments.chunk_size is None else 'source-row chunks'),
                  file=sys.stderr, flush=True)
            if workers == 1 and len(jobs) > 1:
                print('CESS: tasks will run sequentially; use -j N to process files/chunks '
                      'concurrently within available CPU and memory resources.', file=sys.stderr, flush=True)
        if not arguments.manifest:
            print(json.dumps(summary, indent=2))
            if arguments.dry_run:
                return 0
            output.mkdir(parents=True, exist_ok=True)
            return _run_jobs(jobs, summary, workers,
                             threads_per_worker=arguments.threads_per_worker)

        from cess.io.manifest import snapshot, expected_product, verify_product, write_manifest, manifest_lock
        manifest_path = arguments.manifest.expanduser().resolve()
        # A resume preview is strictly read-only, including no lock sidecar or
        # output-directory creation. Actual execution holds the helper's lock.
        lock = nullcontext() if arguments.dry_run and arguments.resume else manifest_lock(manifest_path)
        with lock:
            summary['manifest_version'] = 1
            summary['snapshot'] = snapshot(config, options, mode, arguments.seed, jobs,
                                           output_layout=arguments.output_layout)
            if arguments.resume:
                pending = _resume_plan(manifest_path, summary, jobs, verify_product, expected_product)
            else:
                if manifest_path.exists():
                    raise FileExistsError('Manifest already exists: {}'.format(manifest_path))
                if any(path.exists() or path.is_symlink()
                       for path in (expected_product(job) for job in jobs)):
                    raise FileExistsError('An expected output already exists; a new run never adopts existing products')
                pending = jobs
                summary['results'] = [_pending_result(job) for job in jobs]
                summary['pending_jobs'] = len(jobs)
            print(json.dumps(summary, indent=2), flush=True)
            if arguments.dry_run and arguments.resume:
                return 0
            write_manifest(manifest_path, summary, replace=arguments.resume)
            if arguments.dry_run or not pending:
                return 0
            output.mkdir(parents=True, exist_ok=True)
            return _run_jobs(pending, summary, min(workers, len(pending)),
                persist=lambda: write_manifest(manifest_path, summary, replace=True),
                verify_product=verify_product,
                threads_per_worker=arguments.threads_per_worker)
    except (ValueError, OSError, KeyError) as exc:
        print("CESS: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
