/*
 * NeonCROSS: Vectorized Implementation of the Post-Quantum Signature Algorithm CROSS
 * Copyright (c) 2025 Hanyu Wei et al.
 * Licensed under the Apache License, Version 2.0; see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

.macro pmp_reduction a0, mod, a1, hX, wX
 ushr \a1\wX, \a0\wX, #7
 and \a0\hX, \a0\hX, \mod\hX
 add \a0\wX, \a0\wX, \a1\wX
.endm

.align 4
.global fz_inf_w_by_fz_matrix_lv5
.global _fz_inf_w_by_fz_matrix_lv5
fz_inf_w_by_fz_matrix_lv5:
_fz_inf_w_by_fz_matrix_lv5:
    mov w6, #0x007f
    dup v13.8h, w6
    mov w6, #0
    dup v0.8h, w6
    dup v1.8h, w6
    dup v2.8h, w6
    dup v3.8h, w6
    dup v4.8h, w6
    dup v5.8h, w6
    dup v6.8h, w6
    dup v7.8h, w6
    // acc res
    // v0.8h, v1.8h, v2.8h, v3.8h, v4.8h, v5.8h, v6.8h, v7.8h

    mov w6, #6
matvec_mul_lv5_loop:
    ld1 {v12.8h}, [x1], #16   // vec

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[0]
    mla v1.8h,  v9.8h, v12.h[0]
    mla v2.8h, v10.8h, v12.h[0]
    mla v3.8h, v11.8h, v12.h[0]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[0]
    mla v5.8h,  v9.8h, v12.h[0]
    mla v6.8h, v10.8h, v12.h[0]
    mla v7.8h, v11.8h, v12.h[0]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[1]
    mla v1.8h,  v9.8h, v12.h[1]
    mla v2.8h, v10.8h, v12.h[1]
    mla v3.8h, v11.8h, v12.h[1]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[1]
    mla v5.8h,  v9.8h, v12.h[1]
    mla v6.8h, v10.8h, v12.h[1]
    mla v7.8h, v11.8h, v12.h[1]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[2]
    mla v1.8h,  v9.8h, v12.h[2]
    mla v2.8h, v10.8h, v12.h[2]
    mla v3.8h, v11.8h, v12.h[2]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[2]
    mla v5.8h,  v9.8h, v12.h[2]
    mla v6.8h, v10.8h, v12.h[2]
    mla v7.8h, v11.8h, v12.h[2]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[3]
    mla v1.8h,  v9.8h, v12.h[3]
    mla v2.8h, v10.8h, v12.h[3]
    mla v3.8h, v11.8h, v12.h[3]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[3]
    mla v5.8h,  v9.8h, v12.h[3]
    mla v6.8h, v10.8h, v12.h[3]
    mla v7.8h, v11.8h, v12.h[3]

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h
    pmp_reduction v5, v13, v15, .16b, .8h
    pmp_reduction v6, v13, v15, .16b, .8h
    pmp_reduction v7, v13, v15, .16b, .8h

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[4]
    mla v1.8h,  v9.8h, v12.h[4]
    mla v2.8h, v10.8h, v12.h[4]
    mla v3.8h, v11.8h, v12.h[4]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[4]
    mla v5.8h,  v9.8h, v12.h[4]
    mla v6.8h, v10.8h, v12.h[4]
    mla v7.8h, v11.8h, v12.h[4]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[5]
    mla v1.8h,  v9.8h, v12.h[5]
    mla v2.8h, v10.8h, v12.h[5]
    mla v3.8h, v11.8h, v12.h[5]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[5]
    mla v5.8h,  v9.8h, v12.h[5]
    mla v6.8h, v10.8h, v12.h[5]
    mla v7.8h, v11.8h, v12.h[5]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[6]
    mla v1.8h,  v9.8h, v12.h[6]
    mla v2.8h, v10.8h, v12.h[6]
    mla v3.8h, v11.8h, v12.h[6]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[6]
    mla v5.8h,  v9.8h, v12.h[6]
    mla v6.8h, v10.8h, v12.h[6]
    mla v7.8h, v11.8h, v12.h[6]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[7]
    mla v1.8h,  v9.8h, v12.h[7]
    mla v2.8h, v10.8h, v12.h[7]
    mla v3.8h, v11.8h, v12.h[7]
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64
    mla v4.8h,  v8.8h, v12.h[7]
    mla v5.8h,  v9.8h, v12.h[7]
    mla v6.8h, v10.8h, v12.h[7]
    mla v7.8h, v11.8h, v12.h[7]
    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h
    pmp_reduction v5, v13, v15, .16b, .8h
    pmp_reduction v6, v13, v15, .16b, .8h
    pmp_reduction v7, v13, v15, .16b, .8h

    sub w6, w6, #1
    cbnz w6, matvec_mul_lv5_loop

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h
    pmp_reduction v5, v13, v15, .16b, .8h
    pmp_reduction v6, v13, v15, .16b, .8h
    pmp_reduction v7, v13, v15, .16b, .8h
    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h
    pmp_reduction v5, v13, v15, .16b, .8h
    pmp_reduction v6, v13, v15, .16b, .8h
    pmp_reduction v7, v13, v15, .16b, .8h

    uzp1 v8.16b, v0.16b, v1.16b
    uzp1 v9.16b, v2.16b, v3.16b
    uzp1 v10.16b, v4.16b, v5.16b
    uzp1 v11.16b, v6.16b, v7.16b
    st1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x0]

    ret


.align 4
.global fz_inf_w_by_fz_matrix_lv3
.global _fz_inf_w_by_fz_matrix_lv3
fz_inf_w_by_fz_matrix_lv3:
_fz_inf_w_by_fz_matrix_lv3:
    mov w6, #0x007f
    dup v13.8h, w6
    mov w6, #0
    dup v0.8h, w6
    dup v1.8h, w6
    dup v2.8h, w6
    dup v3.8h, w6
    dup v4.8h, w6
    // acc res
    // v0.8h, v1.8h, v2.8h, v3.8h, v4.8h, v5.8h, v6.8h, v7.8h

    mov w6, #5
matvec_mul_lv3_loop:
    ld1 {v12.8h}, [x1], #16   // vec

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[0]
    mla v1.8h,  v9.8h, v12.h[0]
    mla v2.8h, v10.8h, v12.h[0]
    mla v3.8h, v11.8h, v12.h[0]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[0]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[1]
    mla v1.8h,  v9.8h, v12.h[1]
    mla v2.8h, v10.8h, v12.h[1]
    mla v3.8h, v11.8h, v12.h[1]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[1]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[2]
    mla v1.8h,  v9.8h, v12.h[2]
    mla v2.8h, v10.8h, v12.h[2]
    mla v3.8h, v11.8h, v12.h[2]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[2]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[3]
    mla v1.8h,  v9.8h, v12.h[3]
    mla v2.8h, v10.8h, v12.h[3]
    mla v3.8h, v11.8h, v12.h[3]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[3]

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[4]
    mla v1.8h,  v9.8h, v12.h[4]
    mla v2.8h, v10.8h, v12.h[4]
    mla v3.8h, v11.8h, v12.h[4]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[4]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[5]
    mla v1.8h,  v9.8h, v12.h[5]
    mla v2.8h, v10.8h, v12.h[5]
    mla v3.8h, v11.8h, v12.h[5]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[5]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[6]
    mla v1.8h,  v9.8h, v12.h[6]
    mla v2.8h, v10.8h, v12.h[6]
    mla v3.8h, v11.8h, v12.h[6]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[6]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[7]
    mla v1.8h,  v9.8h, v12.h[7]
    mla v2.8h, v10.8h, v12.h[7]
    mla v3.8h, v11.8h, v12.h[7]
    ld1 {v5.8h}, [x2], #16
    mla v4.8h,  v5.8h, v12.h[7]

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h

    sub w6, w6, #1
    cbnz w6, matvec_mul_lv3_loop
    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h
    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v4, v13, v15, .16b, .8h

    uzp1 v8.16b, v0.16b, v1.16b
    uzp1 v9.16b, v2.16b, v3.16b
    xtn v10.8b, v4.8h
    st1 {v8.16b, v9.16b}, [x0], #32
    st1 {v10.8b}, [x0]

    ret

.align 4
.global fz_inf_w_by_fz_matrix_lv1
.global _fz_inf_w_by_fz_matrix_lv1
fz_inf_w_by_fz_matrix_lv1:
_fz_inf_w_by_fz_matrix_lv1:
    mov w6, #0x007f
    dup v13.8h, w6
    mov w6, #0
    dup v0.8h, w6
    dup v1.8h, w6
    dup v2.8h, w6
    dup v3.8h, w6
    dup v4.8h, w6
    // acc res
    // v0.8h, v1.8h, v2.8h, v3.8h, v4.8h, v5.8h, v6.8h, v7.8h

    mov w6, #3
matvec_mul_lv1_loop:
    ld1 {v12.8h}, [x1], #16   // vec

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[0]
    mla v1.8h,  v9.8h, v12.h[0]
    mla v2.8h, v10.8h, v12.h[0]
    mla v3.8h, v11.8h, v12.h[0]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[1]
    mla v1.8h,  v9.8h, v12.h[1]
    mla v2.8h, v10.8h, v12.h[1]
    mla v3.8h, v11.8h, v12.h[1]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[2]
    mla v1.8h,  v9.8h, v12.h[2]
    mla v2.8h, v10.8h, v12.h[2]
    mla v3.8h, v11.8h, v12.h[2]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[3]
    mla v1.8h,  v9.8h, v12.h[3]
    mla v2.8h, v10.8h, v12.h[3]
    mla v3.8h, v11.8h, v12.h[3]

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[4]
    mla v1.8h,  v9.8h, v12.h[4]
    mla v2.8h, v10.8h, v12.h[4]
    mla v3.8h, v11.8h, v12.h[4]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[5]
    mla v1.8h,  v9.8h, v12.h[5]
    mla v2.8h, v10.8h, v12.h[5]
    mla v3.8h, v11.8h, v12.h[5]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[6]
    mla v1.8h,  v9.8h, v12.h[6]
    mla v2.8h, v10.8h, v12.h[6]
    mla v3.8h, v11.8h, v12.h[6]

    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x2], #64    // mat
    mla v0.8h,  v8.8h, v12.h[7]
    mla v1.8h,  v9.8h, v12.h[7]
    mla v2.8h, v10.8h, v12.h[7]
    mla v3.8h, v11.8h, v12.h[7]

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    sub w6, w6, #1
    cbnz w6, matvec_mul_lv1_loop

    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h
    pmp_reduction v0, v13, v15, .16b, .8h
    pmp_reduction v1, v13, v15, .16b, .8h
    pmp_reduction v2, v13, v15, .16b, .8h
    pmp_reduction v3, v13, v15, .16b, .8h

    uzp1 v8.16b, v0.16b, v1.16b
    uzp1 v9.16b, v2.16b, v3.16b
    st1 {v8.16b, v9.16b}, [x0]

    ret
