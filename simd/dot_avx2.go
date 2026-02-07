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

static float DotProductAVX2(const float* a, const float* b, size_t n) {
	__m256 sum = _mm256_setzero_ps();
	size_t i = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 va = _mm256_loadu_ps(a + i);
		__m256 vb = _mm256_loadu_ps(b + i);
		sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
	}
	float s = horizontal_sum_m256(sum);
	for (; i < n; i++) s += a[i] * b[i];
	return s;
}
*/
import "C"

import "unsafe"

func dotProductAVX2(a, b []float32) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return float64(C.DotProductAVX2(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(n),
	))
}
