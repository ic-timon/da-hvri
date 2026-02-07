package indexer

import (
	"github.com/ic-timon/da-hvri/simd"
	"math/rand"
)

const (
	kMeansK      = 2
	kMeansRounds = 8
)

// SplitLeaf splits a full leaf with K-means K=2 and returns the new InternalNode.
func SplitLeaf(leaf *LeafNode, pool *Pool) *InternalNode {
	cfg := leaf.cfg.OrDefault()
	if leaf.VectorCount() < cfg.SplitThreshold {
		return nil
	}
	vpb := cfg.VectorsPerBlock
	// 收集所有向量与 id
	vecs := make([][]float32, leaf.vectorCount)
	ids := make([]uint64, leaf.vectorCount)
	idx := 0
	for b := 0; b < len(leaf.blocks); b++ {
		d := leaf.blocks[b].Data()
		for s := 0; s < vpb && idx < leaf.vectorCount; s++ {
			vec := make([]float32, BlockDim)
			copy(vec, d[s*BlockDim:(s+1)*BlockDim])
			vecs[idx] = vec
			ids[idx] = leaf.ids[idx]
			idx++
		}
	}
	// K-means K=2，5~10 轮
	assign := kMeans2(vecs, kMeansRounds)
	// 创建 2 个子叶子
	left := NewLeafNode(pool, cfg)
	right := NewLeafNode(pool, cfg)
	for i, a := range assign {
		if a == 0 {
			left.Add(pool, vecs[i], ids[i])
		} else {
			right.Add(pool, vecs[i], ids[i])
		}
	}
	internal := NewInternalNode()
	internal.AddChild(left)
	internal.AddChild(right)
	return internal
}

// kMeans2 对 vectors 做 K=2 聚类，返回每个向量的簇标签 (0 或 1)
func kMeans2(vectors [][]float32, rounds int) []int {
	n := len(vectors)
	if n < 2 {
		return make([]int, n)
	}
	assign := make([]int, n)
	// 随机初始化中心
	c0 := copyVec(vectors[rand.Intn(n)])
	c1 := copyVec(vectors[rand.Intn(n)])
	for r := 0; r < rounds; r++ {
		// 分配
		for i, v := range vectors {
			d0 := simd.DotProduct(v, c0)
			d1 := simd.DotProduct(v, c1)
			if d0 >= d1 {
				assign[i] = 0
			} else {
				assign[i] = 1
			}
		}
		// 更新中心
		var sum0, sum1 []float32
		var cnt0, cnt1 int
		for i, v := range vectors {
			if assign[i] == 0 {
				if sum0 == nil {
					sum0 = make([]float32, BlockDim)
				}
				for j := range v {
					sum0[j] += v[j]
				}
				cnt0++
			} else {
				if sum1 == nil {
					sum1 = make([]float32, BlockDim)
				}
				for j := range v {
					sum1[j] += v[j]
				}
				cnt1++
			}
		}
		if cnt0 > 0 {
			for j := range sum0 {
				c0[j] = sum0[j] / float32(cnt0)
			}
		}
		if cnt1 > 0 {
			for j := range sum1 {
				c1[j] = sum1[j] / float32(cnt1)
			}
		}
	}
	return assign
}

