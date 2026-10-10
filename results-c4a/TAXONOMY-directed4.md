# Failure taxonomy

32 attempts, levels [4], 1 session(s): 20261010-203420

| failure mode | what it means | L4 | total | share |
|---|---|---|---|---|
| other: AssertionError | an exception no rule above names yet | 12 | 12 | 38% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) | 5 | 5 | 16% |
| no tiling: K over 128 | contracted more than one tile's worth in one nc_matmul | 4 | 4 | 12% |
| NaN or Inf | read an uninitialised tile | 4 | 4 | 12% |
| no tiling: M or N over the limit | K is chunked, but the output dimensions are not | 4 | 4 | 12% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error | 1 | 1 | 3% |
| no tiling: partition over 128 | one tile for the whole tensor | 1 | 1 | 3% |
| tile/slice size mismatch | source and destination hold different element counts | 1 | 1 | 3% |

## How the failure moved, round to round

7 repair steps: 2 repeated the same failure (29%), 5 changed it, and 2 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| no tiling: partition over 128 | no tiling: K over 128 | 1 |
| no tiling: K over 128 | NaN or Inf | 1 |
| NaN or Inf | no tiling: M or N over the limit | 1 |
| no tiling: M or N over the limit | wrong buffer | 1 |
| wrong buffer | other: AssertionError | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 1 | 390 | 1581 | 0 | 0 | 0 | 500 | 702 | 536 |
| repair | 6 | 0 | 0 | 1622 | 854 | 0 | 197 | 818 | 565 |
| repair+ledger | 1 | 0 | 0 | 2143 | 648 | 977 | 264 | 1257 | 641 |

## Line-located feedback

The checker quoted the failing line on 28 of 28 attempts that raised (100%).

## One real checker message per failure mode

**result does not fit its tile** (level 4, round 0, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised ValueError: cannot reshape array of size 32768 into shape (128,512) Your code does not call reshape, and removing one will not help. This is raised inside the NKI call on the failing line: its result has 32768 elements and the destination tile you gave it has shape (128, 512), which holds 65536. The destination has the wrong shape. For nisa.nc_matmul with stationary of shape (K, M) and moving of shape (K, N), dst must have shape (M, N). Allocate

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

**wrong buffer** (level 4, round 0, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data between on-chip buffers, sbuf and psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output. The failing line is line 25 of your kernel: `nisa.tensor_copy(dst=out, 

**tile/slice size mismatch** (level 4, round 0, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised ValueError: shape mismatch: value array of shape (65536,) could not be broadcast to indexing result of shape (16384,) The failing line is a call to nisa.tensor_copy, not an assignment. Its result has 65536 elements and its destination `sbuf_stationary` holds 16384. The destination and what goes into it must have IDENTICAL shapes. `sbuf_stationary` was allocated for something else and is being reused here: do not reuse it. Allocate a new tile for

**no tiling: K over 128** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 The contraction dimension K is 256 and one nc_matmul can only contract 128. Split K into chunks of 128 and accumulate: allocate ONE psum tile OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so the partial products add up there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write part

**NaN or Inf** (level 4, round 2, reward 0.50)

> 0 of 4 shapes passed. On K=128 M=128 N=512: NON-FINITE OUTPUT: 65536 NaN and 0 Inf, first at (0, 0). Every one of the 65536 output elements is NaN. Your kernel passes accumulate=True to nisa.nc_matmul. Remove that argument. With it the result here is NaN: a newly allocated psum tile holds NaN, and the product is added to it. Without it, calling nisa.nc_matmul into the same psum tile once per chunk already adds the chunks together. Change nothing else.

**no tiling: M or N over the limit** (level 4, round 3, reward 0.75)

> 2 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 One nisa.nc_matmul call accepts at most 128 on the stationary operand's free dimension (M) and at most 512 on the moving operand's free dimension (N); you passed 256 on the stationary one. Keep your loop over K as it is and add loops over the output: split M into chunks of at most 128 and N into chunks of at most 512. For each (M chunk, N chunk) pair, allocate

**other: AssertionError** (level 4, round 5, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm The failing line is line 53 of your kernel: `nisa.dma_copy(dst=out[m0:m0+chunk_M, n0:n0+chunk_N], src=psum)`. That is the line to change. For reference, the real signature is nisa.dma_copy(dst: 'NkiTensor', src: 'NkiTensor', priority=None, oob_mode=<oob_mode.error: 0>, dge_mode=<dge_mode.unknown: 0>, engine=<engine.unknown: 0>, name=None

