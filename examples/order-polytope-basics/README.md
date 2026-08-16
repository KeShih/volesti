## Compilation
Build the example by running the following commands in this directory.

```bash
cmake . -DLP_SOLVE=_PATH_TO_LIB_FILE
make
```  
You have to specify the path to liblpsolve55.so/dll/dylib.  
For example: -DLP_SOLVE=/usr/lib/lpsolve/liblpsolve55.so

## Usage:
```bash
 ./order_polytope poset_data.txt
```  
where `poset_data.txt` is the file containing the poset data as follows:
- first line of file tells the number of elements in the poset data
- next `m` lines represent the order relations of the poset as a pair of indices, for example:  
a line containing `i j` means A<sub>i</sub> <= A<sub>j</sub>. ( 0 <= `i`, `j` < n)
    
## Counting linear extensions in the log domain

`log_linear_extensions.cpp` estimates `log e(P) = log vol(O(P)) + lgamma(n+1)`
with Gaussian cooling whose stage ratios are accumulated in the log domain
(`logsumexp(log w) - log N`), on a selectable order-polytope Gaussian HMC
backend (spherical event queue by default; the generic CDHR sampler via
`options.use_generic_cdhr`).  It prints the per-stage cooling diagnostics
(annealing parameters, log-ratios, sample counts, weight CV^2) alongside the
estimate, its repeated-run standard error, and the containment-violation
count.  Build it like `order_polytope` above; it needs no input file.
