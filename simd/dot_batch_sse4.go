//go:build amd64 && cgo

package simd

/*
#cgo CFLAGS: -msse4.1 -O3
#include <smmintrin.h>
#include <stddef.h>

static float horizontal_sum_m128(__m128 v) {
	v = _mm_hadd_ps(v, v);
	v = _mm_hadd_ps(v, v);
	return _mm_cvtss_f32(v);
}

void DotProductBatchFlatPrefetchSSE4(const float* query, const float* data, int n, double* results) {
	const size_t dim = 512;
	for (int i = 0; i < n; i++) {
		if (i + 1 < n) {
			_mm_prefetch((const char*)(data + (i + 1) * dim), _MM_HINT_T0);
		}
		__m128 sum = _mm_setzero_ps();
		size_t j = 0;
		for (; j + 4 <= dim; j += 4) {
			__m128 vq = _mm_loadu_ps(query + j);
			__m128 vd = _mm_loadu_ps(data + i * dim + j);
			sum = _mm_add_ps(sum, _mm_mul_ps(vq, vd));
		}
		float s = horizontal_sum_m128(sum);
		for (; j < dim; j++) {
			s += query[j] * data[i * dim + j];
		}
		results[i] = (double)s;
	}
}
*/
import "C"

import "unsafe"

func dotProductBatchFlatSSE4(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	C.DotProductBatchFlatPrefetchSSE4(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(n),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	return results
}
