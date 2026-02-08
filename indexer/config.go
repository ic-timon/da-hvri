package indexer

// Config holds index parameters.
type Config struct {
	VectorsPerBlock   int     // vectors per block, default 64
	SplitThreshold    int     // leaf split threshold, default 512
	SearchWidth       int     // multi-path search width, default 3
	PruneEpsilon      float64 // prune branches with score < maxScore - epsilon, default 0.1
	UseOffheap        bool    // use C.malloc for blocks (requires CGO), reduces GC pressure
	PersistPath       string  // non-empty and file exists: NewTree auto LoadFrom (mmap); read-only tree
	SearchPoolWorkers int     // when >0, enables single-tree search pool (recommend NumCPU) for mmap throttling
}

// DefaultConfig returns the default configuration.
func DefaultConfig() *Config {
	return &Config{
		VectorsPerBlock: 64,
		SplitThreshold:  512,
		SearchWidth:     3,
		PruneEpsilon:    0.1,
	}
}

// OrDefault returns DefaultConfig if c is nil, otherwise normalizes c.
func (c *Config) OrDefault() *Config {
	if c == nil {
		return DefaultConfig()
	}
	if c.VectorsPerBlock <= 0 {
		c.VectorsPerBlock = 64
	}
	if c.SplitThreshold <= 0 {
		c.SplitThreshold = 512
	}
	if c.SearchWidth <= 0 {
		c.SearchWidth = 3
	}
	if c.PruneEpsilon < 0 {
		c.PruneEpsilon = 0.1
	}
	return c
}
