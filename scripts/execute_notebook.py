"""Execute every notebook code cell with the active Python environment.

Default: nbclient/Jupyter kernel. --in-process: ordered IPython execution for
containers without local socket support. Both run all code, including training.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import nbformat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--in-process", action="store_true")
args = parser.parse_args()
nb_path = Path("notebooks/pipeline_analitico.ipynb")
nb = nbformat.read(nb_path, as_version=4)

if args.in_process:
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    original_publish = shell.display_pub.publish
    original_format = shell.displayhook.write_format_data
    outputs = []

    class StreamCapture(io.TextIOBase):
        def __init__(self, name):
            self.name_ = name

        def write(self, text):
            if text:
                if (
                    outputs
                    and outputs[-1].output_type == "stream"
                    and outputs[-1].name == self.name_
                ):
                    outputs[-1].text += text
                else:
                    outputs.append(
                        nbformat.v4.new_output("stream", name=self.name_, text=text)
                    )
            return len(text)

        def flush(self):
            pass

    def publish(data, metadata=None, **kwargs):
        outputs.append(
            nbformat.v4.new_output("display_data", data=data, metadata=metadata or {})
        )

    def implicit(data, metadata=None):
        publish(data, metadata)

    shell.display_pub.publish = publish
    shell.displayhook.write_format_data = implicit
    count = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            outputs = []
            with (
                contextlib.redirect_stdout(StreamCapture("stdout")),
                contextlib.redirect_stderr(StreamCapture("stderr")),
            ):
                result = shell.run_cell(cell.source, store_history=False)
            cell.execution_count = count
            cell.outputs = outputs
            error = result.error_before_exec or result.error_in_exec
            nbformat.write(nb, nb_path)
            if error:
                raise RuntimeError(f"Falló la celda {count}") from error
            print(f"Celda {count} ejecutada", flush=True)
    finally:
        shell.display_pub.publish = original_publish
        shell.displayhook.write_format_data = original_format
    nb.metadata["execution_mode"] = (
        "sequential IPython in-process; all code cells executed"
    )
else:
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    # A globally registered "python3" kernel can point outside the active venv.
    # A temporary kernelspec binds the entire run to this exact interpreter.
    with tempfile.TemporaryDirectory(prefix="aerolife-kernel-") as directory:
        spec = Path(directory) / "aerolife"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "AEROLIFE active environment", "language": "python",
        }), encoding="utf-8")

        class ActiveEnvironmentKernel(AsyncKernelManager):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.kernel_spec_manager = KernelSpecManager(kernel_dirs=[directory])

        NotebookClient(
            nb, timeout=3600, kernel_name="aerolife",
            kernel_manager_class=ActiveEnvironmentKernel,
            resources={"metadata": {"path": "."}},
        ).execute()
    nb.metadata["execution_mode"] = "nbclient Jupyter kernel; all code cells executed"
nbformat.write(nb, nb_path)
print("Notebook ejecutado sin celdas de error.", flush=True)
