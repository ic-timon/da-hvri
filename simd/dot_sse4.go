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

static float DotProductSSE4(const float* a, const float* b, size_t n) {
	__m128 sum = _mm_setzero_ps();
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		__m128 va = _mm_loadu_ps(a + i);
		__m128 vb = _mm_loadu_ps(b + i);
		__m128 prod = _mm_mul_ps(va, vb);
		sum = _mm_add_ps(sum, prod);
	}
	float s = horizontal_sum_m128(sum);
	for (; i < n; i++) s += a[i] * b[i];
	return s;
}
*/
import "C"

import "unsafe"

func dotProductSSE4(a, b []float32) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return float64(C.DotProductSSE4(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(n),
	))
}
