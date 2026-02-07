//go:build amd64 && cgo

package simd

/*
#cgo CFLAGS: -mavx2 -O3
#include <immintrin.h>
#include <stddef.h>

static float horizontal_sum_m256(__m256 v) {
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 lo = _mm256_extractf128_ps(v, 0);
	__m128 sum4 = _mm_add_ps(hi, lo);
	sum4 = _mm_hadd_ps(sum4, sum4);
	sum4 = _mm_hadd_ps(sum4, sum4);
	return _mm_cvtss_f32(sum4);
}

void DotProductBatchFlatPrefetchAVX2(const float* query, const float* data, int n, double* results) {
	const size_t dim = 512;
	for (int i = 0; i < n; i++) {
		if (i + 1 < n) {
			_mm_prefetch((const char*)(data + (i + 1) * dim), _MM_HINT_T0);
		}
		__m256 sum = _mm256_setzero_ps();
		size_t j = 0;
		for (; j + 8 <= dim; j += 8) {
			__m256 vq = _mm256_loadu_ps(query + j);
			__m256 vd = _mm256_loadu_ps(data + i * dim + j);
			sum = _mm256_add_ps(sum, _mm256_mul_ps(vq, vd));
		}
		float s = horizontal_sum_m256(sum);
		for (; j < dim; j++) {
			s += query[j] * data[i * dim + j];
		}
		results[i] = (double)s;
	}
}
*/
import "C"

import "unsafe"

func dotProductBatchFlatAVX2(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	C.DotProductBatchFlatPrefetchAVX2(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(n),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	return results
}
