//go:build arm64 && cgo

package simd

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>

static float DotProductNEON(const float* a, const float* b, size_t n) {
	float32x4_t sum0 = vdupq_n_f32(0.0f);
	float32x4_t sum1 = vdupq_n_f32(0.0f);
	float32x4_t sum2 = vdupq_n_f32(0.0f);
	float32x4_t sum3 = vdupq_n_f32(0.0f);
	size_t i = 0;
	for (; i + 16 <= n; i += 16) {
		float32x4_t va0 = vld1q_f32(a + i);
		float32x4_t vb0 = vld1q_f32(b + i);
		sum0 = vmlaq_f32(sum0, va0, vb0);
		float32x4_t va1 = vld1q_f32(a + i + 4);
		float32x4_t vb1 = vld1q_f32(b + i + 4);
		sum1 = vmlaq_f32(sum1, va1, vb1);
		float32x4_t va2 = vld1q_f32(a + i + 8);
		float32x4_t vb2 = vld1q_f32(b + i + 8);
		sum2 = vmlaq_f32(sum2, va2, vb2);
		float32x4_t va3 = vld1q_f32(a + i + 12);
		float32x4_t vb3 = vld1q_f32(b + i + 12);
		sum3 = vmlaq_f32(sum3, va3, vb3);
	}
	float s = vaddvq_f32(sum0) + vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3);
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
