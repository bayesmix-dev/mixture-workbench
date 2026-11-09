# Python interface

This standard-library-only wrapper launches the Julia command-line sampler and
reads its CSV output. First instantiate the Julia environment as described in the
repository README. The wrapper does not compile model descriptions or generate
sampling algorithms.

From the repository root:

```sh
python3 interfaces/python/mixture_workbench.py \
  --config path/to/config.toml \
  --data path/to/data.csv \
  --output results/python-run
```

Use `--julia /path/to/julia` to select an executable. Paths may contain spaces.
`--entrypoint path/to/implemented-run.jl` selects another implemented
script using the same protocol; relative entrypoints resolve against the
repository root. The default is `bin/run.jl`. The callable API accepts the same
option as `entrypoint=`.
For Python code, add `interfaces/python` to your import path:

```python
import sys
sys.path.insert(0, "interfaces/python")
from mixture_workbench import run_sampler, read_output

fit = run_sampler("path/to/config.toml", "path/to/data.csv", "results/python-run")
print(fit.draws[:3])
print(fit.diagnostics)
print(fit.metadata)
print(fit.parameters)
# Read a completed run later, without launching Julia:
fit = read_output("results/python-run")
```

`fit.draws` and `fit.diagnostics` are lists of dictionaries. `fit.allocations` is
an optional list of dictionaries, and `fit.metadata` is a string dictionary.
Draws and allocations have numeric values. Component labels are local to each
draw; comparing component number 1 across draws is not a relabeling procedure.

Samplers that retain component parameters may also write `parameters.csv`.
The wrapper reads this as `fit.parameters`, or `None` when the file is absent,
including runs produced before this table was introduced. Its required columns
are `chain`, `iteration`, `cluster` (positive integers) and `theta` (a finite
float). `fit.metadata["parameter_name"]` describes `theta`; the implemented
examples use `mean` or `log_rate`. Join parameters to allocations using
`chain`, `iteration` and `cluster`. An optional integer `gem_index` records an
ordered stick-breaking index where the sampler uses one; it does not identify
the same component across draws. Empty tables remain empty lists.

The configuration must select an implemented model and supported settings.
Configuration and data validation belong to the Julia engine. A sampler error
raises `subprocess.CalledProcessError`; the command-line wrapper forwards its
nonzero exit status. This is a synchronous wrapper: interrupting a Python caller
does not provide a managed background-job or checkpointing API.
