package indexer

import "sync"

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
