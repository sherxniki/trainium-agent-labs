# Failure taxonomy

36 attempts, levels [3, 4], 1 session(s): 20261010-202354

| failure mode | what it means | L3 | L4 | total | share |
|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 20 | 2 | 22 | 61% |
| no tiling: M or N over the limit | K is chunked, but the output dimensions are not | 4 |  | 4 | 11% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error | 4 |  | 4 | 11% |
| other: AssertionError | an exception no rule above names yet | 4 |  | 4 | 11% |
| no tiling: partition over 128 | one tile for the whole tensor |  | 2 | 2 | 6% |

## How the failure moved, round to round

7 repair steps: 3 repeated the same failure (43%), 4 changed it, and 0 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| tile/slice size mismatch | no tiling: M or N over the limit | 1 |
| no tiling: M or N over the limit | result does not fit its tile | 1 |
| result does not fit its tile | tile/slice size mismatch | 1 |
| tile/slice size mismatch | other: AssertionError | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 2 | 390 | 1581 | 0 | 0 | 0 | 503 | 702 | 341 |
| repair | 5 | 0 | 0 | 960 | 1054 | 0 | 203 | 689 | 349 |
| repair+ledger | 2 | 0 | 0 | 1122 | 964 | 814 | 270 | 1003 | 348 |

## Line-located feedback

The checker quoted the failing line on 36 of 36 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=8192, dst=65536 The destination has shape (128, 512) but the piece copied into it has shape (128, 64), which is smaller. nisa.dma_copy needs IDENTICAL shapes and cannot fill part of a tile. Allocate a separate tile for this copy with exactly the source's shape: nl.ndarray((128, 64), dtype=..., buffer=nl.sbuf). If one tile is being used for two different ten

**no tiling: M or N over the limit** (level 3, round 1, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: Matmul stationary free dimension 512 exceeds gemm_stationary_fmax=128 One nisa.nc_matmul call accepts at most 128 on the stationary operand's free dimension (M) and at most 512 on the moving operand's free dimension (N); you passed 512 on the stationary one. Keep your loop over K as it is and add loops over the output: split M into chunks of at most 128 and N into chunks of at most 512. For each (M chunk, N chunk) pair, allocate a

**result does not fit its tile** (level 3, round 2, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised ValueError: cannot reshape array of size 32768 into shape (128,512) Your code does not call reshape, and removing one will not help. This is raised inside the NKI call on the failing line: its result has 32768 elements and the destination tile you gave it has shape (128, 512), which holds 65536. The destination has the wrong shape. For nisa.nc_matmul with stationary of shape (K, M) and moving of shape (K, N), dst must have shape (M, N). Allocate 

**other: AssertionError** (level 3, round 7, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm The failing line is line 22 of your kernel: `nisa.dma_copy(dst=out, src=psum_tile)`. That is the line to change. For reference, the real signature is nisa.dma_copy(dst: 'NkiTensor', src: 'NkiTensor', priority=None, oob_mode=<oob_mode.error: 0>, dge_mode=<dge_mode.unknown: 0>, engine=<engine.unknown: 0>, name=None). Pass every argument by 

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

