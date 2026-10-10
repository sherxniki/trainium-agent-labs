# Failure taxonomy

68 attempts, levels [2, 3, 4], 1 session(s): 20261010-193145

| failure mode | what it means | L2 | L3 | L4 | total | share |
|---|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 4 | 15 | 2 | 21 | 31% |
| wrong numbers: core arithmetic | most elements wrong | 16 |  |  | 16 | 24% |
| NaN or Inf | read an uninitialised tile |  |  | 15 | 15 | 22% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error |  | 4 |  | 4 | 6% |
| no tiling: K over 128 | contracted more than one tile's worth in one nc_matmul |  |  | 4 | 4 | 6% |
| 1-D tile | allocated an SBUF/PSUM tile with one dimension |  | 3 |  | 3 | 4% |
| no tiling: partition over 128 | one tile for the whole tensor |  |  | 2 | 2 | 3% |
| other: AssertionError | an exception no rule above names yet |  | 1 |  | 1 | 1% |
| invented API name | called an nl/nisa function that does not exist |  | 1 |  | 1 | 1% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) |  |  | 1 | 1 | 1% |

## How the failure moved, round to round

14 repair steps: 9 repeated the same failure (64%), 5 changed it, and 1 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| tile/slice size mismatch | wrong numbers: core arithmetic | 1 |
| 1-D tile | result does not fit its tile | 1 |
| result does not fit its tile | tile/slice size mismatch | 1 |
| no tiling: partition over 128 | no tiling: K over 128 | 1 |
| no tiling: K over 128 | NaN or Inf | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 3 | 455 | 1581 | 0 | 0 | 0 | 505 | 725 | 291 |
| repair | 8 | 0 | 0 | 1089 | 826 | 0 | 199 | 640 | 381 |
| repair+ledger | 6 | 0 | 0 | 1151 | 579 | 434 | 266 | 767 | 376 |

## Line-located feedback

The checker quoted the failing line on 37 of 37 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 2, round 0, reward 0.30)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=384, dst=512 The destination has shape (128, 4) but the piece copied into it has shape (32, 12), which is smaller. nisa.dma_copy needs IDENTICAL shapes and cannot fill part of a tile. Allocate a separate tile for this copy with exactly the source's shape: nl.ndarray((32, 12), dtype=..., buffer=nl.sbuf). If one tile is being used for two different tenso

**wrong numbers: core arithmetic** (level 2, round 1, reward 0.50)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NUMERICAL MISMATCH: worst error 4.49 of the output's RMS (1.01), tolerance 0.02.
  at index (24, 9): expected +2.62617, got -1.9109
  81.8% of elements are outside tolerance
  most elements are wrong, so this is the core arithmetic or the operand layout, not an edge case

**1-D tile** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) The shape on the failing line has one dimension and a tile needs two. Putting a 1 in front is only right when the data really is a single row. Otherwise give the tile the full shape of the data it will hold: a tile that receives the result of nisa.nc_matmul needs shape (M, N), where stationary is (K, M) and moving is (K, N); a tile that receives a c

**result does not fit its tile** (level 3, round 1, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised ValueError: cannot reshape array of size 32768 into shape (1,64) Your code does not call reshape, and removing one will not help. This is raised inside the NKI call on the failing line: its result has 32768 elements and the destination tile you gave it has shape (1, 64), which holds 64. The destination has the wrong shape. For nisa.nc_matmul with stationary of shape (K, M) and moving of shape (K, N), dst must have shape (M, N). Allocate the desti

**other: AssertionError** (level 3, round 3, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: tensor_copy engine must be one of [vector, scalar, gpsimd, unknown], got <enum 'engine'> The failing line is line 20 of your kernel: `nisa.tensor_copy(dst=sbuf_lhsT, src=psum, engine=nisa.engine)`. That is the line to change. For reference, the real signature is nisa.tensor_copy(dst: 'NkiTensor', src: 'NkiTensor', engine=<engine.unknown: 0>, name=None). Pass every argument by keyword, in those roles.

**invented API name** (level 3, round 3, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AttributeError: type object 'engine' has no attribute 'default' The failing line is line 20 of your kernel: `nisa.tensor_copy(dst=sbuf_lhsT, src=psum, engine=nisa.engine.default)`. That is the line to change. For reference, the real signature is nisa.tensor_copy(dst: 'NkiTensor', src: 'NkiTensor', engine=<engine.unknown: 0>, name=None). Pass every argument by keyword, in those roles.

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

**no tiling: K over 128** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 The contraction dimension K is 256 and one nc_matmul can only contract 128. Split K into chunks of 128 and accumulate: allocate ONE psum tile OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so the partial products add up there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write part

**NaN or Inf** (level 4, round 2, reward 0.50)

> 0 of 4 shapes passed. On K=128 M=128 N=512: NON-FINITE OUTPUT: 65536 NaN and 0 Inf, first at (0, 0). Every one of the 65536 output elements is non-finite, which means the array you return was never written: no line copies a result into it. This is not about initialising a tile, and no function for that exists. After the computation, copy the result into the array you return with nisa.dma_copy(dst=<that array>[...], src=<an sbuf tile>). A psum tile cannot be copied to HBM directly: move it into a

**wrong buffer** (level 4, round 2, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data between on-chip buffers, sbuf and psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output. The failing line is line 46 of your kernel: `nisa.tensor_copy(dst=out, 

