# DA-HVRI: Density-Adaptive Hierarchical Vector Routing Index

[English](README.md) | [中文](README.zh-CN.md)

> **Density-Adaptive Hierarchical Vector Routing Index** — An embedded vector search engine for Go, combining design principles from Milvus, FAISS, and Elasticsearch, optimized for single-machine high-concurrency scenarios.

[![Go 1.21+](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![AVX-512](https://img.shields.io/badge/AVX--512-Accelerated-green)]()
[![P99](https://img.shields.io/badge/mmap_32concurrent_P99-13.47ms-blue)]()
[![mmap](https://img.shields.io/badge/mmap_persist-QPS_4x+-orange)]()

---

## Why DA-HVRI?

In RAG, KAG, and multimodal retrieval, **P99 latency** of vector indexes often determines the user experience ceiling. Common issues with generic open-source libraries under single-machine high concurrency:

- **Goroutine explosion**: Hundreds of goroutines spawned per request; scheduling overhead and cache thrashing drive P99 to 150ms+
- **GC tail**: Heap pressure triggers stop-the-world; P99 collapses randomly
- **Lock contention**: Multiple goroutines compete for the same channel or map; throughput fails to scale linearly

DA-HVRI addresses these with systematic optimizations. At **50k vectors, 32 concurrency**:

| Metric | Value |
|--------|-------|
| **P99 latency** | **13.47 ms** (mmap single tree, 32 concurrent) |
| **QPS** | **8,923** (mmap single tree, 32 concurrent) |
| **P99/P50** | **1.73** (low tail) |
| **Goroutine count** | **17** (stable, no explosion) |
| **mmap persist** | **~4× search QPS vs heap** (contiguous blocks, cache-friendly) |

---

## Design Inspiration

DA-HVRI draws from mainstream vector search systems:

| Project | Inspiration |
|---------|-------------|
| **Milvus / Knowhere** | IVF partitioning + AVX-512 hardware acceleration, multi-core sharding |
| **FAISS** | Centroid routing, nprobe-style multi-path search |
| **Elasticsearch** | Dynamic sharding, off-heap to avoid GC pressure |

Unlike distributed solutions, DA-HVRI targets **single-machine deployment without GPU**, maximizing CPU cache and physical core affinity. It fits well as a local search engine for KAG/RAG applications.

---

## Comparison with PQ and HNSW

DA-HVRI competes with **PQ (Product Quantization)** and **HNSW (Hierarchical Navigable Small World)**. All three target approximate nearest neighbor search but differ in design and use cases:

| Dimension | DA-HVRI | PQ | HNSW |
|-----------|---------|-----|------|
| **Index structure** | Density-adaptive tree (K-means split) | Product quantization codebook + inverted | Multi-layer graph |
| **Build** | Online incremental, no pretraining | Requires codebook training (offline) | Online incremental |
| **Data adaptivity** | Structure evolves with density | Fixed codebook, retrain on distribution shift | Fixed graph, sensitive to distribution |
| **Memory** | Raw vectors + centroids (tunable) | Very low (compressed codes) | Higher (graph + vectors) |
| **Query path** | Tree routing + leaf scan | Table lookup + residual | Graph traversal (multi-hop) |
| **Concurrent P99** | **28.97 ms** (Worker Pool + local queues) | Affected by locks/scheduling | Non-deterministic traversal, common tail |
| **Go ecosystem** | Pure Go + optional CGO | Mostly C++/Python bindings | Mostly C++/Rust bindings |
| **Embedded** | Single binary, no external deps | Needs pretrained codebook | Needs graph structure |

### When to choose DA-HVRI

- **RAG/KAG document stores**: 10k–200k vectors, incremental writes, stable P99
- **Go stack**: Index and business logic in the same language, no FFI
- **Single-machine embedded**: No separate vector service, index embedded in app process
- **Latency determinism**: P99/P50 ratio < 2, no random tail spikes

### When to choose PQ

- **Very large scale**: Millions of vectors, compression-first storage
- **Offline batch**: Index built once, no incremental updates

### When to choose HNSW

- **Recall-first**: Require high recall, acceptable higher latency variance
- **Existing integrations**: Milvus, Qdrant, etc. already integrate HNSW

---

## Core Features

### Density-Adaptive Hierarchical Tree

DA-HVRI’s core data structure: the index evolves with data density, without pre-specified cluster count or depth.

#### Structure

```
                    ┌─────────────────┐
                    │   Root (node)   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            ┌───────────────┐  ┌───────────────┐
            │ InternalNode  │  │ InternalNode   │  ← Centroid routing
            │  (2 children) │  │  (2 children) │
            └───────┬───────┘  └───────┬───────┘
                    │                  │
        ┌───────────┼───────────┐      │
        ▼           ▼           ▼      ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ LeafNode │ │ LeafNode │ │ LeafNode │ │ LeafNode │  ← Leaf layer
  │ Block[]  │ │ Block[]  │ │ Block[]  │ │ Block[]  │    Contiguous blocks
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

- **InternalNode**: Internal node with centroid vectors for routing
- **LeafNode**: Leaf node with DataBlocks, up to 64 vectors of 512 dims per block (configurable)
- **Block**: Contiguous memory block, heap or off-heap (C.malloc)

#### Split mechanism

When leaf vector count reaches `SplitThreshold` (default 512), **K-means K=2** split is triggered:

1. Collect all vectors and chunkIDs in the leaf
2. Initialize 2 centers randomly, iterate 8 rounds for clustering
3. Assign vectors to left/right sub-leaves by cluster label
4. Atomically replace the old leaf with a new InternalNode (left + right children)

Query path is lock-free: root and children use `atomic.Pointer[Node]`; after split, the read path sees a consistent snapshot.

#### Query flow: vector routing

1. Query vector enters at root
2. If InternalNode: compute dot products with centroids, pick best child (or multiple by PruneEpsilon)
3. If LeafNode: batch dot products in leaf blocks (AVX-512), return Top-K
4. Multi-path: each level can explore Top-SearchWidth children, deduplicate at leaves, merge Top-K

Structure adapts to density: sparse regions stay shallow, dense regions split into deeper levels.

---

### Multi-path search and adaptive pruning

- Top-K children per level (`SearchWidth`), balancing recall and latency
- `PruneEpsilon` pruning: only explore branches with `score >= maxScore - ε`

### Hardware acceleration

- **AVX-512** dot product and batch prefetch (`_mm_prefetch`), 10–30% faster leaf scan
- Conditional build: `amd64 && cgo` enables it (Windows/Linux); otherwise pure Go
- **Runtime**: CGO-built binary must run on an AVX-512-capable CPU, or use `CGO_ENABLED=0`

### Low GC impact

- **Off-heap** (C.malloc) for vector blocks; HeapSys reduced ~60%
- `sync.Pool` reuse for seen map; no allocation on P99 path

### High-concurrency determinism

- **Worker Pool**: Fixed workers instead of 16 goroutines per request; goroutines drop from ~512 to 17
- **Local queues**: Per-worker channels, route by `shardIdx % nWorkers` to avoid global contention
- **Physical core affinity**: Worker count `max(nShards, NumCPU/2)` to avoid hyperthread contention

---

## Performance Benchmarks

**Environment**: Windows / Go 1.21 / 512-dim vectors

### 32-concurrent (50k vectors, mmap single tree, recommended)

Run: `go run ./bench -stage c -offheap`

| Concurrency | QPS | P50(ms) | P99(ms) | P99/P50 |
|-------------|-----|---------|---------|---------|
| 1 | 1,693 | 0.52 | 1.72 | 3.28 |
| 4 | 4,212 | 1.00 | 2.28 | 2.28 |
| 8 | 6,435 | 1.00 | 3.89 | 3.89 |
| 16 | 8,065 | 1.12 | 6.84 | 6.10 |
| 32 | **8,923** | 1.12 | **13.47** | 12.05 |

### 32-concurrent (50k vectors, 16 shards heap)

Run: `go run ./bench -stage c -shards 16 -offheap`

| Concurrency | QPS | P50(ms) | P99(ms) | P99/P50 |
|-------------|-----|---------|---------|---------|
| 1 | 800 | 1.00 | 2.45 | 2.44 |
| 4 | 975 | 4.18 | 6.28 | 1.50 |
| 8 | 1,136 | 7.07 | 11.78 | 1.67 |
| 16 | 1,439 | 10.88 | 19.22 | 1.77 |
| 32 | 1,911 | 16.76 | 25.69 | 1.53 |

### Capacity (100k / 200k vectors, mmap)

Run: `go run ./bench -stage b`

| Scale | Search P50 | Search P99 | HeapSys |
|-------|-------------|-------------|---------|
| 100k | ~0 ms | ~0.6 ms | ~700 MB |
| 200k | ~0.5 ms | ~1.6 ms | ~2.8 GB |

### CGO vs no CGO

Without CGO: pure Go dot product and heap memory. With CGO: AVX-512 and off-heap. (50k vectors, 16 shards)

| Concurrency | Metric | No CGO | CGO |
|-------------|--------|--------|-----|
| 1 | QPS | 701 | **887** |
| 1 | P50(ms) | 1.50 | **1.00** |
| 32 | QPS | 873 | **1,668** |
| 32 | P50(ms) | 37.31 | **18.32** |
| 32 | P99(ms) | 42.14 | **33.67** |

CGO yields ~1.9× QPS and lower P50/P99. Without CGO it still builds and runs, suitable when GCC is unavailable or for cross-compilation.

### Heap vs mmap persist (stage d)

Run: `go run ./bench -stage d`

| Mode | QPS | P50 | P99 | Ratio |
|------|-----|-----|-----|-------|
| Heap | ~1,600 | ~8 ms | ~22 ms | baseline |
| **mmap persist** | **~7,900** | **~1.1 ms** | **~7.7 ms** | **~4.9×** |

mmap stores blocks contiguously in the file; sequential access improves CPU prefetch and cache locality over heap-scattered blocks. Use `NewTreeFromFile` or `cfg.PersistPath` for serving.

---

## Quick Start

### Dependencies

- Go 1.21+
- CGO (optional, for AVX-512 and off-heap)
- **Embedding model**: Development/benchmark uses BGE 512-dim (e.g. bge-small-zh-v1.5)
- **Windows**: MinGW-w64 or MSYS2, `gcc` in PATH
- **Linux**: build-essential (GCC) or Clang, `gcc`/`clang` in PATH

> **CGO runtime**: With CGO enabled, the binary must run on an x86_64 CPU with **AVX-512** support, or it may crash with SIGILL. On hosts without AVX-512, build with `CGO_ENABLED=0` to fall back to pure Go.

### Build and run

**Windows**

```powershell
# Enable CGO (recommended)
$env:CGO_ENABLED = "1"
go build -o bench.exe ./bench

# Benchmark (stage: a param tune | b capacity | c high concurrency | d heap vs mmap)
.\bench.exe -stage c -shards 16 -offheap
.\bench.exe -stage d   # Compare mmap vs heap search performance
```

**Linux**

```bash
# With CGO (requires amd64 + AVX-512 CPU)
CGO_ENABLED=1 go build -o bench ./bench

# Without CGO (any amd64, no AVX-512 required)
CGO_ENABLED=0 go build -o bench ./bench

# Benchmark (stage: a|b|c|d)
./bench -stage c -shards 16 -offheap
./bench -stage d   # Compare mmap vs heap search performance
```

---

## Usage

### 1. Index type

| Type | Use case | Create |
|------|----------|--------|
| **Single tree** | Small scale (< 10k), low concurrency | `indexer.NewTree(cfg)` |
| **Sharded index** | Medium/large scale, high concurrency | `indexer.NewShardedIndex(cfg, 16)` |

Sharded index routes vectors by `chunkID % nShards` to 16 trees; search queries all in parallel and merges results.

### 2. Config and creation

#### DefaultConfig

`DefaultConfig()` returns tuned defaults for 10k–200k vectors and moderate concurrency:

| Parameter | Default | Role | Tuning |
|-----------|---------|------|--------|
| **VectorsPerBlock** | 64 | vectors per block; ~128KB, fits L2; AVX-512 prefetches by block | 32 for less memory, 128 for fewer blocks; 64 is cache-friendly |
| **SplitThreshold** | 512 | leaf split threshold; triggers K=2 split | 128/256: deeper tree; 1024: shallower, lower latency |
| **SearchWidth** | 3 | children per level in multi-path | Higher: more recall, higher latency; 3 is a balance |
| **PruneEpsilon** | 0.1 | only enter branches with `score ≥ maxScore - ε` | 0.05: stricter; 0.2: looser |
| **UseOffheap** | false | use C.malloc for blocks | **Set true for production** (requires CGO) |
| **PersistPath** | "" | when non-empty and file exists, NewTree auto LoadFrom (mmap) | set when loading index for serving |

Recommended: `DefaultConfig()` + `UseOffheap = true` + `nShards = 16`.

```go
import "github.com/ic-timon/da-hvri/indexer"

// Default (recommended)
cfg := indexer.DefaultConfig()
cfg.UseOffheap = true

// Or custom
cfg := &indexer.Config{
    VectorsPerBlock: 64,
    SplitThreshold:  512,
    SearchWidth:     3,
    PruneEpsilon:    0.1,
    UseOffheap:      true,
    PersistPath:     "",   // set path for serving; NewTree auto mmap when file exists
}

// Single tree
tree := indexer.NewTree(cfg)

// Sharded index (recommended)
idx := indexer.NewShardedIndex(cfg, 16)
```

### 3. Insert vectors

Vectors must be **512-dim**, **L2-normalized** `[]float32`. `chunkID` is your chunk identifier, returned as-is in results.

```go
// Single insert
vec := []float32{...}  // len == 512
ok := idx.Add(vec, uint64(chunkID))
if !ok {
    // invalid dimension or insert failed
}

// Batch insert (vec from embedding model or service)
for i, vec := range vectors {
    if !idx.Add(vec, uint64(ids[i])) {
        log.Fatalf("add failed id %d", ids[i])
    }
}
```

### 4. Search API

| Method | Description |
|--------|-------------|
| `Search(query, k)` | Single-path search, lowest latency |
| `SearchMultiPath(query, k)` | Multi-path search, higher recall |

```go
// queryVec is 512-dim L2-normalized
results := idx.SearchMultiPath(queryVec, 5)

for _, r := range results {
    fmt.Printf("chunk %d, score %.4f\n", r.ChunkID, r.Score)
}
```

`SearchResult`:

```go
type SearchResult struct {
    ChunkID uint64   // chunk ID from Add
    Score   float64  // cosine similarity (dot product for L2-normalized)
}
```

### 5. Index-only (no embedder)

If you already have 512-dim vectors:

```go
cfg := indexer.DefaultConfig()
cfg.UseOffheap = true
idx := indexer.NewShardedIndex(cfg, 16)

// Insert
for i, vec := range myVectors {
    idx.Add(vec, uint64(myIDs[i]))
}

// Search
results := idx.SearchMultiPath(queryVec, 10)
```

### 6. Persistence and loading (~4× search QPS with mmap)

Build phase uses heap tree; after SaveToAtomic, the serving process loads via mmap (default; contiguous blocks yield better cache locality than heap):

```go
// Build process: Add -> SaveToAtomic
cfg := indexer.DefaultConfig()
tree := indexer.NewTree(cfg)
for i, vec := range vectors {
    tree.Add(vec, uint64(ids[i]))
}
if err := tree.SaveToAtomic("/path/to/index.bin"); err != nil {
    log.Fatal(err)
}

// Serving process: NewTreeFromFile or cfg.PersistPath + NewTree (mmap default)
tree, err := indexer.NewTreeFromFile("/path/to/index.bin", cfg)
if err != nil { log.Fatal(err) }
defer tree.ClosePersisted()
results := tree.SearchMultiPath(queryVec, 5)
```

Or use `cfg.PersistPath`:

```go
cfg.PersistPath = "/path/to/index.bin"
tree := indexer.NewTree(cfg)  // auto LoadFrom (mmap) if file exists
defer tree.ClosePersisted()
```

mmap is the default load path; blocks are contiguous in the file for better cache locality. Use `indexer.AppendTo(path, vecs, ids, cfg)` for incremental updates. Call `ClosePersisted()` on exit to release the mmap.

### 7. Notes

- **Vector dimension**: Must be 512
  - **AVX-512**: `__m512` processes 16 float32 per step; 512/16=32 iterations, no scalar tail
  - **L1/L2-friendly**: 2KB per vector, 64 vectors ~128KB per block within L2; prefetch with contiguous layout
  - Matches `indexer.BlockDim` and common embedding models (e.g. BGE 512-dim)
- **Normalization**: Vectors must be L2-normalized or dot product is not cosine similarity
- **CGO**: `UseOffheap=true` requires CGO; falls back to heap when CGO is disabled
- **Concurrency**: `Add` and `SearchMultiPath` are safe to call concurrently; tree read path is lock-free

---

## Project structure

```
.
├── indexer/          # Index core
│   ├── tree.go       # Dynamic descending tree
│   ├── shard.go      # Sharded index + Worker Pool
│   ├── persist.go    # SaveToAtomic / LoadFrom / NewTreeFromFile / AppendTo
│   ├── block_mmap.go # mmap blocks (read-only, default for search)
│   ├── store/        # Persist format and mmap store
│   └── ...
├── simd/             # AVX-512 dot product (CGO)
└── bench/            # Benchmarks (stage a|b|c|d)
```

---

## Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| VectorsPerBlock | 64 | Vectors per block |
| SplitThreshold | 512 | Leaf split threshold |
| SearchWidth | 3 | Multi-path search width |
| PruneEpsilon | 0.1 | Adaptive pruning threshold |
| UseOffheap | false | Enable C.malloc |
| PersistPath | "" | Serving load path; NewTree auto mmap |

---

## License

MIT
