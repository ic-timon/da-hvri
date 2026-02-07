// Package gen 提供压测用随机向量生成
package gen

import (
	"math"
	"math/rand"
)

// RandomVectors 生成 n 个 dim 维 L2 归一化随机向量，用于不依赖 embedder 的纯索引压测
func RandomVectors(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			x := rng.Float32()
			v[j] = x
			norm += float64(x * x)
		}
		norm = math.Sqrt(norm)
		if norm < 1e-9 {
			v[0] = 1
			norm = 1
		}
		for j := 0; j < dim; j++ {
			v[j] /= float32(norm)
		}
		out[i] = v
	}
	return out
}
