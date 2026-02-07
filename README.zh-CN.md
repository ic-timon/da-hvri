# DA-HVRI：基于密度自适应的层级化向量路由索引

[English](README.md) | [中文](README.zh-CN.md)

> **Density-Adaptive Hierarchical Vector Routing Index** — 专为 Go 打造的嵌入式向量检索引擎，融合 Milvus、FAISS、Elasticsearch 的设计精髓，针对单机高并发场景深度优化。

[![Go 1.21+](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![AVX-512](https://img.shields.io/badge/AVX--512-超级加速-green)]()
[![P99](https://img.shields.io/badge/32并发_P99-28.97ms-blue)]()

---

## 为什么选择 DA-HVRI？

在 RAG、KAG 与多模态检索场景中，向量索引的 **P99 延迟** 往往决定用户体验上限。通用开源库在单机高并发下常见问题：

- **协程爆炸**：每请求 spawn 数百 goroutine，调度开销与缓存抖动导致 P99 飙升至 150ms+
- **GC 长尾**：堆内存压力触发 stop-the-world，P99 随机崩塌
- **锁竞争**：多 goroutine 争抢同一 channel 或 map，吞吐量难以线性扩展

DA-HVRI 针对上述痛点进行了系统性优化，在 **50k 向量、32 并发** 下实现：

| 指标 | 数值 |
|------|------|
| **P99 延迟** | **28.97 ms** |
| **QPS** | **1,910** |
| **P99/P50** | **1.73**（长尾极低） |
| **Goroutine 数** | **17**（稳定，无爆炸） |

---

## 设计灵感与对标

DA-HVRI 的设计理念借鉴了业界主流向量检索系统的核心思想：

| 项目 | 借鉴点 |
|------|--------|
| **Milvus / Knowhere** | IVF 分区 + AVX-512 硬件加速，多核分片架构 |
| **FAISS** | 聚类中心路由、`nprobe` 风格的多路径检索 |
| **Elasticsearch** | 动态分片、Off-heap 规避 GC 压力 |

与分布式方案不同，DA-HVRI 面向 **无GPU单机部署** 场景，极致利用 CPU 缓存与物理核亲和性，更适合作为 KAG/RAG 应用的本地检索引擎。

---

## 与 PQ、HNSW 的对比

DA-HVRI 的核心竞品是 **PQ（Product Quantization）** 与 **HNSW（Hierarchical Navigable Small World）**。三者均面向近似最近邻检索，但设计哲学与适用场景不同：

| 维度 | DA-HVRI | PQ | HNSW |
|------|---------|-----|------|
| **索引结构** | 密度自适应树（K-means 分裂） | 乘积量化码本 + 倒排 | 多层图 |
| **构建方式** | 在线增量，无预训练 | 需训练码本（离线） | 在线增量 |
| **数据适应性** | 结构随密度自动演化 | 固定码本，分布变化需重训 | 图结构固定，对分布敏感 |
| **内存占用** | 原始向量 + 质心（可控） | 极低（压缩码） | 较高（图 + 向量） |
| **查询路径** | 树路由 + 叶子扫描 | 查表 + 残差计算 | 图遍历（多跳） |
| **并发 P99** | **28.97 ms**（Worker Pool + 本地队列） | 易受锁/调度影响 | 图遍历非确定性，长尾常见 |
| **Go 生态** | 纯 Go + CGO 可选 | 多为 C++/Python 绑定 | 多为 C++/Rust 绑定 |
| **嵌入式部署** | 单二进制，无外部依赖 | 需加载预训练码本 | 需加载图结构 |

### 何时选 DA-HVRI？

- **RAG/KAG 文档库**：10k–200k 向量，增量写入，需稳定 P99
- **Go 技术栈**：希望索引与业务同语言，无 FFI 跨调用
- **单机嵌入式**：无独立向量服务，索引内嵌应用进程
- **延迟确定性**：P99/P50 比值 < 2，拒绝随机长尾

### 何时选 PQ？

- **超大规模**：百万级以上的压缩存储优先
- **离线批处理**：索引一次性构建，不需增量更新

### 何时选 HNSW？

- **召回优先**：对召回率要求极高，可接受更高延迟方差
- **已有成熟实现**：如 Milvus、Qdrant 等已深度集成 HNSW

---

## 核心特性

### 密度自适应下沉树（Density-Adaptive Hierarchical Tree）

DA-HVRI 的核心数据结构，索引结构随数据密度自动演化，无需预先指定聚类数或层级深度。

#### 结构组成

```
                    ┌─────────────────┐
                    │   Root (节点)   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            ┌───────────────┐  ┌───────────────┐
            │ InternalNode  │  │ InternalNode   │  ← 质心路由层
            │  (2 个子节点)  │  │  (2 个子节点)  │
            └───────┬───────┘  └───────┬───────┘
                    │                  │
        ┌───────────┼───────────┐      │
        ▼           ▼           ▼      ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ LeafNode │ │ LeafNode │ │ LeafNode │ │ LeafNode │  ← 叶子层
  │ Block[]  │ │ Block[]  │ │ Block[]  │ │ Block[]  │    连续内存块
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

- **InternalNode**：内部节点，含 2 个质心向量（Centroid），用于路由决策
- **LeafNode**：叶子节点，挂载若干 DataBlock，每块最多 64 个 512 维向量（可配置）
- **Block**：连续内存块，支持堆内存或 Off-heap（C.malloc）

#### 下沉分裂（Split）机制

当叶子节点向量数达到 `SplitThreshold`（默认 512）时，自动触发 **K-means K=2** 分裂：

1. 收集叶子内所有向量与 chunkID
2. 随机初始化 2 个中心，迭代 8 轮进行聚类
3. 按簇标签分配向量到左/右子叶子
4. 原子替换旧叶子为新的 InternalNode（左子节点 + 右子节点）

查询路径无需加锁：根节点与子节点均为 `atomic.Pointer[Node]`，分裂完成后原子替换，读路径始终看到一致快照。

#### 查询流程：向量路由

1. 查询向量进入根节点
2. 若为 InternalNode：计算与各质心的点积，选择得分最高的子节点（或按 PruneEpsilon 进入多个分支）
3. 若为 LeafNode：在叶子块内做批量点积（AVX-512），返回 Top-K
4. 多路径检索：每层可选 Top-SearchWidth 个子节点，叶子去重后合并全局 Top-K

结构随数据密度自动演化：稀疏区域保持浅层，密集区域自动下沉为多层级，实现「密度自适应」。

---

### 多路径检索与自适应剪枝

- 每层选取 Top-K 子节点（`SearchWidth`），平衡召回与延迟
- `PruneEpsilon` 剪枝：仅进入 `score >= maxScore - ε` 的分支，减少无效遍历

### 硬件级加速

- **AVX-512** 点积与批量预取（`_mm_prefetch`），叶子扫描 10–30% 加速
- 条件编译：`amd64 && cgo` 启用（Windows/Linux 通用），否则回退纯 Go
- **运行时**：CGO 构建的二进制需在支持 AVX-512 的 CPU 上运行，否则使用 `CGO_ENABLED=0`

### 零 GC 干扰

- **Off-heap**（`C.malloc`）分配向量块，HeapSys 降低约 60%
- `sync.Pool` 复用 seen map，P99 路径无分配

### 高并发确定性

- **Worker Pool**：常驻 worker 替代每请求 spawn 16 goroutine，goroutine 数从 ~512 压至 17
- **本地队列**：每 worker 独立 channel，按 `shardIdx % nWorkers` 路由，消除全局 channel 竞争
- **物理核亲和**：worker 数 `max(nShards, NumCPU/2)`，规避超线程竞争

---

## 性能基准

**环境**：Windows / Go 1.21 / 512 维向量

### 32 并发压测（50k 向量，16 分片，Off-heap）

| 并发 | QPS | P50(ms) | P99(ms) | P99/P50 |
|------|-----|---------|---------|---------|
| 1 | 837 | 1.01 | 2.52 | 2.49 |
| 4 | 1,088 | 3.61 | 6.06 | 1.68 |
| 8 | 1,230 | 6.63 | 10.34 | 1.56 |
| 16 | 1,430 | 11.08 | 20.40 | 1.84 |
| 32 | **1,910** | 16.72 | **28.97** | **1.73** |

### 容量扩展（100k / 200k 向量）

| 规模 | 搜索 P50 | 搜索 P99 | HeapSys (Off-heap) |
|------|----------|----------|--------------------|
| 100k | ~3.7 ms | ~5 ms | ~404 MB |
| 200k | ~8 ms | ~9 ms | ~804 MB |

### CGO 与 无 CGO 对比

无 CGO 时回退到纯 Go 点积与堆内存，CGO 启用 AVX-512 与 Off-heap，性能差异如下（50k 向量，16 分片）：

| 并发 | 指标 | 无 CGO | CGO |
|------|------|--------|-----|
| 1 | QPS | 701 | **887** |
| 1 | P50(ms) | 1.50 | **1.00** |
| 32 | QPS | 873 | **1,668** |
| 32 | P50(ms) | 37.31 | **18.32** |
| 32 | P99(ms) | 42.14 | **33.67** |

CGO 下 QPS 约可提升 1.9 倍，P50/P99 显著降低。无 CGO 时仍可正常编译运行，适合无 GCC 或交叉编译场景。

---

## 快速开始

### 依赖

- Go 1.21+
- CGO（可选，用于 AVX-512 与 Off-heap）
- **嵌入模型**：开发/压测使用 BGE 512 维（如 bge-small-zh-v1.5）
- **Windows**：MinGW-w64 或 MSYS2，`gcc` 在 `PATH` 中
- **Linux**：`build-essential`（GCC）或 Clang，`gcc`/`clang` 在 `PATH` 中

> **CGO 运行时要求**：启用 CGO 构建时，二进制需在支持 **AVX-512** 的 x86_64 CPU 上运行，否则可能出现 `SIGILL` 崩溃。若部署环境无 AVX-512（如老旧云主机），请使用 `CGO_ENABLED=0` 构建，将回退到纯 Go 实现。

### 构建与运行

**Windows**

```powershell
# 启用 CGO（推荐）
$env:CGO_ENABLED = "1"
go build -o bench.exe ./bench

# 压测
.\bench.exe -stage c -shards 16 -offheap
```

**Linux**

```bash
# 启用 CGO（需 amd64 + AVX-512 CPU）
CGO_ENABLED=1 go build -o bench ./bench

# 无 CGO（任意 amd64，无 AVX-512 要求）
CGO_ENABLED=0 go build -o bench ./bench

# 压测
./bench -stage c -shards 16 -offheap
```

---

## 使用说明

### 1. 索引类型选择

| 类型 | 适用场景 | 创建方式 |
|------|----------|----------|
| **单树** | 小规模（< 10k）、低并发 | `indexer.NewTree(cfg)` |
| **分片索引** | 中大规模、高并发 | `indexer.NewShardedIndex(cfg, 16)` |

分片索引将向量按 `chunkID % nShards` 路由到 16 棵独立树，检索时并行查询后合并，适合 RAG/KAG 中的大规模文档库。

### 2. 配置与创建

#### DefaultConfig 配置说明

`DefaultConfig()` 返回针对本架构调优的默认值，适合 10k–200k 向量、中等并发场景。各参数含义与调优建议：

| 参数 | 默认 | 在本架构下的作用 | 调优建议 |
|------|------|------------------|----------|
| **VectorsPerBlock** | 64 | 每块向量数，块大小 = 64×512×4 ≈ 128KB，落在 L2 缓存范围，AVX-512 批量预取以块为单位 | 32 更省内存、128 更少块数；保持 64 对 cache 最友好 |
| **SplitThreshold** | 512 | 叶子内向量数达此值触发 K=2 分裂，树深度自适应 | 128/256 树更深、召回更精细；1024 树更浅、延迟更低 |
| **SearchWidth** | 3 | 每层进的子节点数，多路径检索 | 增大召回更高、延迟上升；3 为延迟/召回平衡点 |
| **PruneEpsilon** | 0.1 | 仅进入 `score ≥ maxScore - ε` 的分支 | 0.05 更严格剪枝、0.2 更宽松，一般保持默认 |
| **UseOffheap** | false | 为 true 时用 C.malloc 分配块，减少 GC | **生产高并发建议 true**（需 CGO） |

分片索引推荐：`DefaultConfig()` + `UseOffheap = true` + `nShards = 16`。

```go
import "github.com/ic-timon/da-hvri/indexer"

// 默认配置（推荐）
cfg := indexer.DefaultConfig()
cfg.UseOffheap = true  // 生产环境建议开启

// 或自定义
cfg := &indexer.Config{
    VectorsPerBlock: 64,   // 每块向量数
    SplitThreshold:  512,   // 叶子分裂阈值
    SearchWidth:     3,    // 多路径宽度
    PruneEpsilon:    0.1,  // 剪枝阈值
    UseOffheap:      true, // 启用 C.malloc（需 CGO）
}

// 单树（小规模）
tree := indexer.NewTree(cfg)

// 分片索引（推荐）
idx := indexer.NewShardedIndex(cfg, 16)
```

### 3. 插入向量

向量必须为 **512 维**、**L2 归一化** 的 `[]float32`。`chunkID` 为业务侧 chunk 唯一标识，检索结果中会原样返回。

```go
// 单条插入
vec := []float32{...}  // len == 512
ok := idx.Add(vec, uint64(chunkID))
if !ok {
    // 向量维度错误或插入失败
}

// 批量插入（vec 从嵌入模型或服务获取）
for i, vec := range vectors {
    if !idx.Add(vec, uint64(ids[i])) {
        log.Fatalf("添加失败 id %d", ids[i])
    }
}
```

### 4. 检索 API

| 方法 | 说明 |
|------|------|
| `Search(query, k)` | 单路径检索，每层只走得分最高分支，延迟最低 |
| `SearchMultiPath(query, k)` | 多路径检索，每层进 Top-SearchWidth 分支，召回更高 |

```go
// queryVec 为 512 维 L2 归一化向量
results := idx.SearchMultiPath(queryVec, 5)

for _, r := range results {
    fmt.Printf("chunk %d, score %.4f\n", r.ChunkID, r.Score)
}
```

`SearchResult` 结构体：

```go
type SearchResult struct {
    ChunkID uint64   // 插入时传入的 chunk 标识
    Score   float64  // 余弦相似度（点积，因已 L2 归一化）
}
```

### 5. 仅使用索引模块

若你已有 512 维向量（如从其他嵌入服务获取），可直接使用 indexer：

```go
cfg := indexer.DefaultConfig()
cfg.UseOffheap = true
idx := indexer.NewShardedIndex(cfg, 16)

// 插入
for i, vec := range myVectors {
    idx.Add(vec, uint64(myIDs[i]))
}

// 检索
results := idx.SearchMultiPath(queryVec, 10)
```

### 6. 注意事项

- **向量维度**：必须为 512
  - **AVX-512**：`__m512` 一次处理 16 个 float32，512÷16=32 次循环，无余数、无标量尾部，点积全程 SIMD
  - **L1/L2 友好**：单向量 512×4=2KB，可完整放入 L1d（典型 32KB）；每块 64 向量 ≈128KB，落在 L2 范围，预取 `_mm_prefetch` 配合连续布局，减少 cache miss
  - 与 `indexer.BlockDim` 及常见嵌入模型（如 BGE 512 维）一致
- **归一化**：向量需 L2 归一化，否则点积不能表示余弦相似度
- **CGO**：`UseOffheap=true` 需 CGO；禁用 CGO 时自动回退堆内存
- **并发**：`Add` 与 `SearchMultiPath` 可并发调用，树结构读路径无锁

---

## 项目结构

```
.
├── indexer/          # 索引核心
│   ├── tree.go       # 动态下沉树
│   ├── shard.go      # 分片索引 + Worker Pool
│   ├── search_pool.go # 本地队列
│   ├── block.go      # 堆块
│   ├── block_offheap.go  # Off-heap 块（CGO）
│   └── ...
├── simd/             # AVX-512 点积（CGO）
└── bench/            # 压测
```

---

## 配置参数

| 参数 | 默认 | 说明 |
|------|------|------|
| VectorsPerBlock | 64 | 每块向量数 |
| SplitThreshold | 512 | 叶子分裂阈值 |
| SearchWidth | 3 | 多路径搜索宽度 |
| PruneEpsilon | 0.1 | 自适应剪枝阈值 |
| UseOffheap | false | 启用 C.malloc |

---

## License

MIT
