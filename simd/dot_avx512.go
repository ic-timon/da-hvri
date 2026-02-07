//go:build amd64 && cgo

package simd

/*
#cgo CFLAGS: -mavx512f -O3
#include <immintrin.h>
#include <stddef.h>

static float DotProductAVX512(const float* a, const float* b, size_t n) {
	__m512 sum = _mm512_setzero_ps();
	size_t i = 0;
	for (; i + 16 <= n; i += 16) {
		__m512 va = _mm512_loadu_ps(a + i);
		__m512 vb = _mm512_loadu_ps(b + i);
		sum = _mm512_fmadd_ps(va, vb, sum);
	}
	float result[16];
	_mm512_storeu_ps(result, sum);
	float s = 0;
	for (int j = 0; j < 16; j++) s += result[j];
	for (; i < n; i++) s += a[i] * b[i];
	return s;
}
*/
import "C"

import "unsafe"

func dotProductAVX512(a, b []float32) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return float64(C.DotProductAVX512(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(n),
	))
}
