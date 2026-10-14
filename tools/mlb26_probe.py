from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
GAME = ROOT / "dist/AnyPS5/games/MLBTheShow26"
SOURCE = Path(os.environ.get("APS5_MLB_SOURCE", r"C:\Users\rsoto\Downloads\[DLPSGAME.COM]-PPSA30630\PPSA30630-app0"))
LOGS = ROOT / "build/mlb26-inspection"


def stage():
    library_directory = GAME / "libs"
    if library_directory.is_junction():
        if library_directory.resolve() != (ROOT / "dist/AnyPS5/libs").resolve():
            raise RuntimeError("Unexpected library junction target")
        library_directory.rmdir()
    if not library_directory.exists():
        shutil.copytree(ROOT / "dist/AnyPS5/libs", library_directory)
    app = GAME / "app0"
    app.mkdir(parents=True, exist_ok=True)
    directories = [(p, app / p.name) for p in SOURCE.iterdir()
                   if p.is_dir() and p.name not in {"PS5", "sce_module", "fakelib", "sce_sys"}]
    (app / "PS5").mkdir(exist_ok=True)
    directories.extend((p, app / "PS5" / p.name) for p in (SOURCE / "PS5").iterdir()
                       if p.is_dir() and p.name != "SCE_MODULE")
    for source, target in directories:
        if target.exists():
            if not target.is_junction() or target.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected resource directory: {target}")
            continue
        command = ["powershell.exe", "-NoProfile", "-Command",
                   "New-Item -ItemType Junction -Path $env:APS5_STAGE_TARGET -Target ([System.Management.Automation.WildcardPattern]::Escape($env:APS5_STAGE_SOURCE)) | Out-Null"]
        environment = dict(os.environ, APS5_STAGE_TARGET=str(target), APS5_STAGE_SOURCE=str(source))
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    shutil.copytree(SOURCE / "sce_sys", app / "sce_sys", dirs_exist_ok=True)
    for source in SOURCE.iterdir():
        if source.is_file():
            target = app / source.name
            if not target.exists():
                shutil.copy2(source, target)
    extracted = LOGS / "source"
    for source in extracted.rglob("*.prx"):
        target = app / source.relative_to(extracted)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print("Resources staged; large asset directories reference the supplied dump.")


def deploy():
    if (GAME / "libs").is_junction():
        raise RuntimeError("Deployment requires MLB's private library directory")
    sources = [ROOT / f"build/core/libs/libs/{name}.prx" for name in
               ("libc", "libkernel", "libSceAudioOut", "libSceAgcDriver", "libSceAgc", "libSceVideoOut")]
    if any(not source.is_file() for source in sources):
        raise RuntimeError("Build all matching runtime libraries before deployment")
    backup = LOGS / "deployments" / str(time.time_ns())
    backup.mkdir(parents=True)
    report = []
    for source in sources:
        target = GAME / "libs" / source.name
        if target.exists():
            shutil.copy2(target, backup / target.name)
    deployed = []
    try:
        for source in sources:
            target = GAME / "libs" / source.name
            shutil.copy2(source, target)
            deployed.append(target)
            report.append({"library": source.stem, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    except OSError:
        for target in deployed:
            if (backup / target.name).is_file():
                shutil.copy2(backup / target.name, target)
        raise
    (backup / "deployment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Deployed matching runtime libraries; backup: {backup}", flush=True)


def launch(seconds, label, game_args, experimental=False, keep_running=False, depth_resolve="sample0", terminal_mips="on", alpha_without_color="on"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("label must contain only letters, digits, underscores and hyphens")
    run = LOGS / "runs" / f"{label}-{time.time_ns()}"
    run.mkdir(parents=True)
    log = run / "game.log"
    pad_file = run / "pad-input.txt"
    pad_file.write_text("", encoding="utf-8")
    environment = dict(os.environ, ANYPS5_GPU="RX 6900 XT", APS5_TRACE_CALLS="all",
                       APS5_PAD_INPUT_FILE=str(pad_file), APS5_DUMP_FRAMES="600",
                       APS5_DUMP_FRAMES_EVERY="300", APS5_DUMP_SCALE="4",
                       APS5_MIXED_SAMPLE_CAPTURE_DIR=str(run), APS5_DUMP_FRAME_DIR=str(run),
                       APS5_DUMP_FAILED_SHADERS="1", APS5_SHADER_DUMP_DIR=str(run))
    environment.pop("APS5_APPROXIMATE_MIXED_SAMPLES", None)
    environment.pop("APS5_EXPERIMENTAL_DEPTH_RESOLVE", None)
    environment.pop("APS5_APPROXIMATE_TERMINAL_MIPS", None)
    environment.pop("APS5_ALPHA_WITHOUT_COLOR", None)
    if experimental:
        environment["APS5_APPROXIMATE_MIXED_SAMPLES"] = "1"
        environment["APS5_EXPERIMENTAL_DEPTH_RESOLVE"] = depth_resolve
        environment["APS5_APPROXIMATE_TERMINAL_MIPS"] = "1" if terminal_mips == "on" else "0"
        environment["APS5_ALPHA_WITHOUT_COLOR"] = "1" if alpha_without_color == "on" else "0"
    ctypes.windll.kernel32.SetErrorMode(0x8003)
    report_file = run / "report.json"
    started = time.monotonic()
    with log.open("wb") as output:
        process = subprocess.Popen([str(GAME / "MLBTheShow26-filtered.exe"), *game_args], cwd=GAME,
                                   env=environment, stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        report = {"pid": process.pid, "log": str(log), "run_directory": str(run),
                  "pad_input": str(pad_file), "experimental": experimental,
                  "depth_resolve": depth_resolve if experimental else "off",
                  "terminal_mips": terminal_mips if experimental else "off",
                  "alpha_without_color": alpha_without_color if experimental else "off",
                  "executable": str(GAME / "MLBTheShow26-filtered.exe"), "arguments": game_args,
                  "exit_code": None, "running": True, "result": "observing"}
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        next_status = 30
        while process.poll() is None and time.monotonic() - started < seconds:
            time.sleep(1)
            elapsed = time.monotonic() - started
            if elapsed >= next_status:
                print(f"Observed {elapsed:.0f}s; process running; {len(list(run.glob('frame_*.bmp')))} frames captured", flush=True)
                next_status += 30
        timed_out = process.poll() is None
        if timed_out and not keep_running:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        report.update(exit_code=process.poll(), running=process.poll() is None,
                      observed_seconds=round(time.monotonic() - started, 1),
                      result="observation_timeout" if timed_out else "exited",
                      stopped_by_probe=timed_out and not keep_running)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print(json.dumps(report), flush=True)
    print("\n".join(line[:1000] for line in lines[-25:]))
    return 2 if timed_out else int(report["exit_code"] != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--label", default="strict")
    parser.add_argument("--game-arg", action="append", default=[])
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--depth-resolve", choices=("off", "clear", "sample0", "min", "max"), default="sample0")
    parser.add_argument("--terminal-mips", choices=("off", "on"), default="on")
    parser.add_argument("--alpha-without-color", choices=("off", "on"), default="on")
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()
    if args.stage:
        stage()
    if args.deploy:
        deploy()
    if args.launch:
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        raise SystemExit(launch(args.seconds, args.label, args.game_arg, args.experimental, args.keep_running, args.depth_resolve, args.terminal_mips, args.alpha_without_color))
