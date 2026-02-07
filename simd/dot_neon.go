//go:build arm64 && cgo

package simd

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>

static float DotProductNEON(const float* a, const float* b, size_t n) {
	float32x4_t sum = vdupq_n_f32(0.0f);
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		float32x4_t va = vld1q_f32(a + i);
		float32x4_t vb = vld1q_f32(b + i);
		sum = vmlaq_f32(sum, va, vb);
	}
	float s = vaddvq_f32(sum);
	for (; i < n; i++) s += a[i] * b[i];
	return s;
}
*/
import "C"

import "unsafe"

func dotProductNEON(a, b []float32) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	return float64(C.DotProductNEON(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(n),
	))
}
