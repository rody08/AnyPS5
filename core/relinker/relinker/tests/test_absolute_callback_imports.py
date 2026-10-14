import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from test_unused_filter_indirect import build, at_offset, CODE, SLOT, rel


def fixture():
    object_base = SLOT - 16
    alias = 0x830
    save = CODE + 0x40
    invoke = CODE + 0x60
    code = at_offset([
        (0, b"\x48\x8d\x3d" + rel(CODE + 7, object_base)
         + b"\xe8" + rel(CODE + 12, save)
         + b"\xe8" + rel(CODE + 17, invoke) + b"\xc3"),
        (0x40, b"\x48\x89\x3d" + rel(save + 7, alias) + b"\xc3"),
        (0x60, b"\x48\x8b\x05" + rel(invoke + 7, alias) + b"\xff\x60\x10"),
    ])
    image = build(code)
    struct.pack_into("<Q", image, 0x708, (1 << 32) | 1)
    return image


def main():
    relinker = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="anyps5-absolute-callback-") as directory:
        work = Path(directory)
        source = work / "callback.elf"
        source.write_bytes(fixture())
        for level in (1, 2):
            for windows in (False, True):
                output = work / f"callback-{level}-{windows}.out"
                arguments = [str(relinker), "--skip-sce-module", "--registry", f"unused-filter={level}"]
                if windows:
                    arguments.append("--windows")
                result = subprocess.run([*arguments, str(source), str(output)],
                                        capture_output=True, text=True, timeout=20)
                assert result.returncode == 0, (level, windows, result.stdout, result.stderr)
                registry = json.loads(output.with_suffix(".registry.json").read_text())
                assert len(registry) == 1, (level, windows, registry)
                assert registry[0]["nid"] == "foo" and registry[0]["targetOffset"].lower() == "0x800", registry
                assert registry[0]["relocationType"] == "R_X86_64_64", registry
    print("Absolute allocator callbacks retained through mutable object aliases")


if __name__ == "__main__":
    main()
