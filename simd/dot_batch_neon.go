//go:build arm64 && cgo

package simd

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>

void DotProductBatchFlatPrefetchNEON(const float* query, const float* data, int n, double* results) {
	const size_t dim = 512;
	for (int i = 0; i < n; i++) {
		if (i + 2 < n) {
			__builtin_prefetch(data + (i + 2) * dim);
		}
		float32x4_t sum0 = vdupq_n_f32(0.0f);
		float32x4_t sum1 = vdupq_n_f32(0.0f);
		float32x4_t sum2 = vdupq_n_f32(0.0f);
		float32x4_t sum3 = vdupq_n_f32(0.0f);
		size_t j = 0;
		for (; j + 16 <= dim; j += 16) {
			float32x4_t vq0 = vld1q_f32(query + j);
			float32x4_t vd0 = vld1q_f32(data + i * dim + j);
			sum0 = vmlaq_f32(sum0, vq0, vd0);
			float32x4_t vq1 = vld1q_f32(query + j + 4);
			float32x4_t vd1 = vld1q_f32(data + i * dim + j + 4);
			sum1 = vmlaq_f32(sum1, vq1, vd1);
			float32x4_t vq2 = vld1q_f32(query + j + 8);
			float32x4_t vd2 = vld1q_f32(data + i * dim + j + 8);
			sum2 = vmlaq_f32(sum2, vq2, vd2);
			float32x4_t vq3 = vld1q_f32(query + j + 12);
			float32x4_t vd3 = vld1q_f32(data + i * dim + j + 12);
			sum3 = vmlaq_f32(sum3, vq3, vd3);
		}
		float s = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);
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
