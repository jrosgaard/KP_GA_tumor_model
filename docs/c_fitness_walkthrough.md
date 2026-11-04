# Learning C through the population fitness evaluator

The starting point is `GA/population_fitness.c`. It implements the same fitness
formula as `GeneticAlgorithm.fitness_func`, with one loop over a population.
The Python function is kept as a readable reference and for correctness tests.

## Build and use it

From the project root, using the Python environment where you run the notebooks:

```sh
python3 -m GA.native_fitness --build
```

This requires a C compiler (`cc`). On macOS the build produces
`GA/_population_fitness.dylib`; on Linux it produces a `.so` file. These are
shared libraries: compiled machine code that Python can load into its process.
They are ignored by Git because each machine should build its own copy.
Rebuild after editing the C source, then restart your notebook kernel: an
already running process can retain the previously loaded machine code.

Both GA configurations in `main.ipynb` now use:

```python
fitness_func=GeneticAlgorithm.fitness_func_batch,
fitness_batch_size=100,
```

The batch size must be greater than one and no larger than `sol_per_pop`.
PyGAD can supply fewer rows when it has cached some candidates' fitnesses.
The evaluator handles any number of rows, including one.
See the [PyGAD batch interface documentation](https://pygad.readthedocs.io/en/stable/fitness_calculation.html#batch-fitness-calculation).

To switch back to the reference implementation, change the callback to
`GeneticAlgorithm.fitness_func` and remove `fitness_batch_size`.

## Follow one call

1. PyGAD gives `fitness_func_batch` a two-dimensional array of genes.
2. The callback resolves the same environment defaults and overrides as before.
3. `GA/native_fitness.py` checks the inputs and prepares contiguous float64 arrays.
4. Python's built-in `ctypes` passes their memory addresses to the compiled C function.
5. C reads each candidate's eight genes, computes the fitness, and writes a result.
6. Python returns the filled NumPy array to PyGAD.

The C library is loaded once per Python process. No compiler or subprocess is
started during fitness evaluation. The model parameters and fitness weights
are packed once per batch, rather than once per candidate.

## The C syntax in this example

```c
const double *genes = population + 8*i;
double s_1 = genes[0]*x + genes[1]*y + genes[2]*z + genes[3];
```

`double` is the floating-point type used for the numerical calculations.
Unlike Python, C requires a declared type for each variable. A semicolon ends
each statement, and braces group statements into a block.

The `*` in a declaration means a pointer: a variable holding a memory address.
Here `genes` points at the first gene of candidate `i`. Adding `8*i` to a
`double` pointer advances by that many doubles, not that many bytes.
`genes[0]` reads the first double at that address. `const` prevents writing to
the genes through this pointer; it does not copy the data.

For a population with three candidates, the buffer looks like:

```text
candidate 0: indices  0..7
candidate 1: indices  8..15
candidate 2: indices 16..23
```

NumPy's contiguous, row-major layout puts each row directly after the previous
one. The wrapper makes a contiguous copy when the original array is a slice
with gaps. C does not know the array shape and does not check array bounds;
the Python shape checks and the C loop limit are therefore essential.

```c
for (size_t i = 0; i < count; ++i) {
    /* Compute candidate i's fitness here. */
}
```

`size_t` is an unsigned integer type designed for sizes and array indices.
The loop initializes `i` to zero, checks `i < count` before each iteration,
and increments `i` afterwards. This is the role of `range(count)` in Python.

```c
fitness_out[i] = c1*immunotherapy - c2*toxicity;
```

This writes directly into an array allocated by Python. The C function returns
`void`, meaning no return value; the output travels through this pointer.
Python keeps all buffers alive during the call. C neither allocates nor frees
them, and does not keep any pointers after the function finishes.

Notice that a square is written `z_pred*z_pred`. Python's `**` operator does not
exist in C, and C's `^` operator means bitwise exclusive OR.

## What calculation is being accelerated?

The first four genes define `s_1`, and the last four define `s_2`. Each is an
affine function of the current state. Negative treatment inputs are clipped to
zero. The evaluator then predicts the next state with one forward-Euler step
and applies the existing immune-response and toxicity weights.

Terms independent of the genes are computed once outside the C loop. The
prediction formula, clipping, defaults, and fitness weights are unchanged.
The RK45 integrator that advances the actual simulation remains in Python.
The current equations do not use `t`, so the C function does not need that input.

The C wrapper requires exactly eight genes and finite inputs, and rejects zero
model denominators. Extremely large finite inputs can still overflow during
arithmetic. As with the reference implementation, choose sensible model ranges.
Floating-point results can differ by tiny rounding amounts, so tests compare
them with a tolerance rather than requiring identical bits.

## Check your changes

With `pytest` and the project dependencies installed:

```sh
python3 -m pytest tests/test_native_fitness.py
```

The tests compile the C library and compare it against the original Python
function for random populations, clipped dosing, parameter overrides, different
state scales, and partial batches. They also check the memory-layout boundary
and run a small actual PyGAD optimization when PyGAD is installed.

As an exercise, trace a candidate whose eight genes are all zero. Both treatment
inputs should be zero. Then trace a candidate with only gene 3 set to 2: this
provides a constant `s_1 = 2`, independently of the current state. Follow how that
changes `x_pred`, the response reward, and the toxicity penalty.
