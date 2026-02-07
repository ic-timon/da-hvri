package indexer

import (
	"runtime"
	"sync"
)

// singleTreeSearchJob 单树检索任务
type singleTreeSearchJob struct {
	query  []float32
	k      int
	result []SearchResult
	wg     sync.WaitGroup
}

// singleTreeSearchPool 单树专用 worker 池，限流并发以降低 P99/P50
type singleTreeSearchPool struct {
	tree *Tree
	jobs chan *singleTreeSearchJob
	wg   sync.WaitGroup
}

func newSingleTreeSearchPool(tree *Tree, nWorkers, bufSize int) *singleTreeSearchPool {
	if nWorkers <= 0 {
		nWorkers = runtime.NumCPU()
	}
	p := &singleTreeSearchPool{
		tree: tree,
		jobs: make(chan *singleTreeSearchJob, bufSize),
	}
	for i := 0; i < nWorkers; i++ {
		p.wg.Add(1)
		go p.worker()
	}
	return p
}

func (p *singleTreeSearchPool) worker() {
	defer p.wg.Done()
	for job := range p.jobs {
		job.result = p.tree.searchMultiPathImpl(job.query, job.k)
		job.wg.Done()
	}
}

func (p *singleTreeSearchPool) Search(query []float32, k int) []SearchResult {
	job := &singleTreeSearchJob{query: query, k: k}
	job.wg.Add(1)
	p.jobs <- job
	job.wg.Wait()
	return job.result
}

func (p *singleTreeSearchPool) Close() {
	close(p.jobs)
	p.wg.Wait()
}

// searchJob 单分片检索任务
type searchJob struct {
	shardIdx int
	query    []float32
	shard    *Tree
	k        int
	results  [][]SearchResult
	wg       *sync.WaitGroup
}

// searchWorkerPool 常驻 worker 池，每 worker 独立 channel
type searchWorkerPool struct {
	chans []chan searchJob
	wg    sync.WaitGroup
}

// newSearchWorkerPool 创建并启动 worker 池
func newSearchWorkerPool(nWorkers, bufSize int) *searchWorkerPool {
	p := &searchWorkerPool{
		chans: make([]chan searchJob, nWorkers),
	}
	for i := 0; i < nWorkers; i++ {
		p.chans[i] = make(chan searchJob, bufSize)
		p.wg.Add(1)
		go p.worker(i)
	}
	return p
}

func (p *searchWorkerPool) worker(idx int) {
	defer p.wg.Done()
	for job := range p.chans[idx] {
		job.results[job.shardIdx] = job.shard.SearchMultiPath(job.query, job.k)
		job.wg.Done()
	}
}

// Submit 按 shardIdx 路由到对应 worker
func (p *searchWorkerPool) Submit(job searchJob) {
	idx := job.shardIdx % len(p.chans)
	p.chans[idx] <- job
}

// Close 关闭池，等待所有 worker 退出
func (p *searchWorkerPool) Close() {
	for i := range p.chans {
		close(p.chans[i])
	}
	p.wg.Wait()
}
