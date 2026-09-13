// Run the built wheel in actual WebAssembly Python, without mounting src/.
import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { loadPyodide } from "pyodide";

const [wheel] = process.argv.slice(2);
if (!wheel || process.argv.length !== 3 || !wheel.endsWith("-py3-none-any.whl")) {
  throw new Error("Usage: node run.mjs /path/to/cunumpy-VERSION-py3-none-any.whl");
}

const pyodide = await loadPyodide();
await pyodide.loadPackage(["micropip", "pytest"]);
pyodide.FS.writeFile(`/tmp/${basename(wheel)}`, await readFile(wheel));
pyodide.globals.set("wheel_uri", `emfs:/tmp/${basename(wheel)}`);
await pyodide.runPythonAsync(`
import micropip
await micropip.install(wheel_uri)
`);
pyodide.FS.mkdirTree("/tests");
pyodide.FS.writeFile(
  "/tests/test_numpy_runtime.py",
  await readFile(new URL("../portable/test_numpy_runtime.py", import.meta.url)),
);
const result = pyodide.runPython(`
import importlib.abc
import importlib.metadata
import sys

assert sys.platform == "emscripten", sys.platform

class ForbidOptionalImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cupy", "pyccel"}:
            self.attempted.append(fullname)
            raise AssertionError(f"CPU execution attempted to import {fullname}")

guard = ForbidOptionalImports()
sys.meta_path.insert(0, guard)
try:
    import cunumpy
    import pytest

    assert cunumpy.numpy_backend and not cunumpy.cupy_backend
    assert cunumpy.__version__ == importlib.metadata.version("cunumpy")
    assert "site-packages" in cunumpy.__file__, cunumpy.__file__
    print(f"Testing cunumpy {cunumpy.__version__} in Python {sys.version}")
    exit_code = int(pytest.main(["-q", "-o", "addopts=", "/tests"]))
    assert not guard.attempted, guard.attempted
    assert not any(name.split(".")[0] in {"cupy", "pyccel"} for name in sys.modules)
finally:
    sys.meta_path.remove(guard)
exit_code
`);
if (result !== 0) {
  throw new Error(`Pyodide pytest failed with exit code ${result}`);
}
