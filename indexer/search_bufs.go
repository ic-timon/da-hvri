package indexer

const (
	scoresBufCap  = 512
	indicesBufCap = 512
	seenBufCap    = 256
)

// seenEntry is a compact (id, score) pair for deduplication.
type seenEntry struct {
	id    uint64
	score float64
}

// seenSlice replaces map[uint64]float64 for cache-friendly linear scan.
type seenSlice []seenEntry

func (s *seenSlice) upsert(id uint64, score float64) {
	for i := range *s {
		if (*s)[i].id == id {
			if score > (*s)[i].score {
				(*s)[i].score = score
			}
			return
		}
	}
	*s = append(*s, seenEntry{id: id, score: score})
}

func (s *seenSlice) reset() {
	*s = (*s)[:0]
}

const maxBatchSize = 16

// workerBufs holds per-worker reusable buffers to avoid cross-core sharing.
type workerBufs struct {
	scores       []float64
	indices      []int
	seen         seenSlice
	seenBatch    []seenSlice // for batch mode
	batchScores  [][]float64 // batchScores[q] for query q in leaf
	batchIndices [][]int     // batchIndices[q] for query q in leaf
}

func newWorkerBufs() *workerBufs {
	return &workerBufs{
		scores:       make([]float64, 0, scoresBufCap),
		indices:      make([]int, 0, indicesBufCap),
		seen:         make(seenSlice, 0, seenBufCap),
		seenBatch:    make([]seenSlice, 0, maxBatchSize),
		batchScores:  make([][]float64, 0, maxBatchSize),
		batchIndices: make([][]int, 0, maxBatchSize),
	}
}

func (b *workerBufs) reset() {
	b.seen.reset()
	b.scores = b.scores[:0]
	b.indices = b.indices[:0]
}

func (b *workerBufs) ensureBatch(n int) {
	for len(b.seenBatch) < n {
		b.seenBatch = append(b.seenBatch, make(seenSlice, 0, seenBufCap))
	}
	for len(b.batchScores) < n {
		b.batchScores = append(b.batchScores, make([]float64, 0, scoresBufCap))
	}
	for len(b.batchIndices) < n {
		b.batchIndices = append(b.batchIndices, make([]int, 0, indicesBufCap))
	}
	for i := 0; i < n; i++ {
		b.seenBatch[i].reset()
	}
}
