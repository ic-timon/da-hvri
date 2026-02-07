//go:build arm64 && cgo

package simd

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>

void DotProductBatchFlatPrefetchNEON(const float* query, const float* data, int n, double* results) {
	const size_t dim = 512;
	for (int i = 0; i < n; i++) {
		if (i + 1 < n) {
			__builtin_prefetch(data + (i + 1) * dim);
		}
		float32x4_t sum = vdupq_n_f32(0.0f);
		size_t j = 0;
		for (; j + 4 <= dim; j += 4) {
			float32x4_t vq = vld1q_f32(query + j);
			float32x4_t vd = vld1q_f32(data + i * dim + j);
			sum = vmlaq_f32(sum, vq, vd);
		}
		float s = vaddvq_f32(sum);
		for (; j < dim; j++) {
			s += query[j] * data[i * dim + j];
		}
		results[i] = (double)s;
	}
}
*/
import "C"

import "unsafe"

func dotProductBatchFlatNEON(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	C.DotProductBatchFlatPrefetchNEON(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(n),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	return results
}
