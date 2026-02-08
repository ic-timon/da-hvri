//go:build amd64 && cgo

package simd

/*
#cgo CFLAGS: -mavx512f -mavx512dq -O3
#include <immintrin.h>
#include <stddef.h>

static float horizontal_sum_m512(__m512 v) {
	__m256 hi = _mm512_extractf32x8_ps(v, 1);
	__m256 lo = _mm512_extractf32x8_ps(v, 0);
	__m256 sum8 = _mm256_add_ps(hi, lo);
	__m128 hi4 = _mm256_extractf128_ps(sum8, 1);
	__m128 lo4 = _mm256_extractf128_ps(sum8, 0);
	__m128 sum4 = _mm_add_ps(hi4, lo4);
	sum4 = _mm_hadd_ps(sum4, sum4);
	sum4 = _mm_hadd_ps(sum4, sum4);
	return _mm_cvtss_f32(sum4);
}

// DotProductBatchFlatPrefetch 对连续内存中的 n 个向量分别与 query 计算点积，循环内预取下一块
void DotProductBatchFlatPrefetch(const float* query, const float* data, int n, double* results) {
	const size_t dim = 512;
	for (int i = 0; i < n; i++) {
		if (i + 2 < n) {
			_mm_prefetch((const char*)(data + (i + 2) * dim), _MM_HINT_T0);
		}
		__m512 sum = _mm512_setzero_ps();
		size_t j = 0;
		for (; j + 16 <= dim; j += 16) {
			__m512 vq = _mm512_loadu_ps(query + j);
			__m512 vd = _mm512_loadu_ps(data + i * dim + j);
			sum = _mm512_fmadd_ps(vq, vd, sum);
		}
		float s = horizontal_sum_m512(sum);
		for (; j < dim; j++) {
			s += query[j] * data[i * dim + j];
		}
		results[i] = (double)s;
	}
}
*/
import "C"

import "unsafe"

func dotProductBatchFlatAVX512(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	C.DotProductBatchFlatPrefetch(
		(*C.float)(unsafe.Pointer(&query[0])),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(n),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	return results
}
