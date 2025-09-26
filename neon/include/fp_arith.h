/*
 * NeonCROSS: Vectorized Implementation of the Post-Quantum Signature Algorithm CROSS
 * Copyright (c) 2025 Hanyu Wei et al.
 * Licensed under the Apache License, Version 2.0; see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include "architecture_detect.h"
#include "csprng_hash.h"
#include "parameters.h"
#include "restr_arith.h"

#define NUM_BITS_P (BITS_TO_REPRESENT(P))

#if defined(RSDP)
#define FPRED_SINGLE(x) (((x) & 0x7F) + ((x) >> 7))
#define FPRED_DOUBLE(x) FPRED_SINGLE(FPRED_SINGLE(x))
#define FPRED_OPPOSITE(x) ((x) ^ 0x7F)
#define FP_DOUBLE_ZERO_NORM(x) (((x) + (((x) + 1) >> 7)) & 0x7F)
#define RESTR_TO_VAL(x) ( (FP_ELEM) (RESTR_G_TABLE >> (8*(uint64_t)(x))) )

#elif defined(RSDPG)
/* Reduction modulo P=509 as shown in:
 * Hacker's Delight, Second Edition, Chapter 10, Figure 10-4
 * Works for integers in the range [0,4294967295] i.e. all uint32_t */
#define FPRED_SINGLE(x) (((x) - (((uint64_t)(x) * 2160140723) >> 40) * P))
#define FPRED_DOUBLE(x) (FPRED_SINGLE(x))
#define FPRED_OPPOSITE(x) (FPRED_SINGLE(P - (x)))
#define FPRED_MERSENNE(x) (((x) & 0x1FF) + ((x) >> 9)*3)
#define FPRED_MERSENNE_DOUBLE(x) FPRED_MERSENNE(FPRED_MERSENNE(x))
/* no redundant zero notation in F_509 */
#define FP_DOUBLE_ZERO_NORM(x) (x)

/* for i in [0,1,2,4,8,16,32,64] RESTR_G_GEN**i mod 509 yields
 * [1, 16, 256, 384, 355, 302, 93, 505]
 * the following is a precomputed-squares S&M, to be optimized into muxed
 * register stored tables */

#define RESTR_G_GEN_1  ((FP_ELEM)RESTR_G_GEN) // 000010000_2
#define RESTR_G_GEN_2  ((FP_ELEM) 256)        // 100000000_2
#define RESTR_G_GEN_4  ((FP_ELEM) 384)        // 110000000_2
#define RESTR_G_GEN_8  ((FP_ELEM) 355)        // 101100011_2
#define RESTR_G_GEN_16 ((FP_ELEM) 302)        // 100101110_2
#define RESTR_G_GEN_32 ((FP_ELEM) 93)         // 001011101_2
#define RESTR_G_GEN_64 ((FP_ELEM) 505)        // 111111001_2

#define FP_ELEM_CMOV(BIT,TRUE_V,FALSE_V)  ( (((FP_ELEM)0 - (BIT)) & (TRUE_V)) | (~((FP_ELEM)0 - (BIT)) & (FALSE_V)) )     // BIT=0, FALSE_V; BIT=1, TRUE_V

/* log reduction, constant time unrolled S&M w/precomputed squares.
 * To be further optimized with muxed register-fitting tables */
static inline
FP_ELEM RESTR_TO_VAL(FP_ELEM x){
    uint32_t res1, res2, res3, res4;
    res1 = ( FP_ELEM_CMOV(((x >> 0) &1),RESTR_G_GEN_1 ,1)) *
           ( FP_ELEM_CMOV(((x >> 1) &1),RESTR_G_GEN_2 ,1)) ;
    res2 = ( FP_ELEM_CMOV(((x >> 2) &1),RESTR_G_GEN_4 ,1)) *
           ( FP_ELEM_CMOV(((x >> 3) &1),RESTR_G_GEN_8 ,1)) ;
    res3 = ( FP_ELEM_CMOV(((x >> 4) &1),RESTR_G_GEN_16,1)) *
           ( FP_ELEM_CMOV(((x >> 5) &1),RESTR_G_GEN_32,1)) ;
    res4 =   FP_ELEM_CMOV(((x >> 6) &1),RESTR_G_GEN_64,1);

    /* Two intermediate reductions necessary:
     *     RESTR_G_GEN_1*RESTR_G_GEN_2*RESTR_G_GEN_4*RESTR_G_GEN_8    < 2^32
     *     RESTR_G_GEN_16*RESTR_G_GEN_32*RESTR_G_GEN_64               < 2^32 */
    return FPRED_SINGLE( FPRED_SINGLE(res1 * res2) * FPRED_SINGLE(res3 * res4) );
}

static FP_ELEM TABLE_RESTR_TO_VAL[2*Z] = {
    1, 16, 256, 24, 384, 36, 67, 54, 355, 81, 278, 376, 417, 55, 371, 337,
    302, 251, 453, 122, 425, 183, 383, 20, 320, 30, 480, 45, 211, 322, 62, 483,
    93, 470, 394, 196, 82, 294, 123, 441, 439, 407, 404, 356, 97, 25, 400, 292,
    91, 438, 391, 148, 332, 222, 498, 333, 238, 245, 357, 113, 281, 424, 167, 127,
    505, 445, 503, 413, 500, 365, 241, 293, 107, 185, 415, 23, 368, 289, 43, 179,
    319, 14, 224, 21, 336, 286, 504, 429, 247, 389, 116, 329, 174, 239, 261, 104,
    137, 156, 460, 234, 181, 351, 17, 272, 280, 408, 420, 103, 121, 409, 436, 359,
    145, 284, 472, 426, 199, 130, 44, 195, 66, 38, 99, 57, 403, 340, 350,
    1, 16, 256, 24, 384, 36, 67, 54, 355, 81, 278, 376, 417, 55, 371, 337,
    302, 251, 453, 122, 425, 183, 383, 20, 320, 30, 480, 45, 211, 322, 62, 483,
    93, 470, 394, 196, 82, 294, 123, 441, 439, 407, 404, 356, 97, 25, 400, 292,
    91, 438, 391, 148, 332, 222, 498, 333, 238, 245, 357, 113, 281, 424, 167, 127,
    505, 445, 503, 413, 500, 365, 241, 293, 107, 185, 415, 23, 368, 289, 43, 179,
    319, 14, 224, 21, 336, 286, 504, 429, 247, 389, 116, 329, 174, 239, 261, 104,
    137, 156, 460, 234, 181, 351, 17, 272, 280, 408, 420, 103, 121, 409, 436, 359,
    145, 284, 472, 426, 199, 130, 44, 195, 66, 38, 99, 57, 403, 340, 350,
};

#endif

#define vmovl_u8_xidx(a, b, idx) \
for(int i = 0; i < idx; i++) \
    a.val[i] = vmovl_u8(b.val[i]); \

#define vmovl_u16_xidx(a, b, idx) \
for(int i = 0; i < idx; i++) \
    a.val[i] = vmovl_u16(b.val[i]); \

#define vmovn_u16_xidx(a, b, idx) \
for(int i = 0; i < idx; i++)  \
    a.val[i] = vmovn_u16(b.val[i]); \

#define vmovn_u32_xidx(a, b, idx) \
for(int i = 0; i < idx; i++)  \
    a.val[i] = vmovn_u32(b.val[i]); \

#define vec_by_scalar_mla_fpred_double_u16_x3(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));\
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));\
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vaddq_u16(vshrq_n_u16(vecr.val[2], 7), vandq_u16(vecr.val[2], neon_p));\
    vecr.val[2] = vaddq_u16(vshrq_n_u16(vecr.val[2], 7), vandq_u16(vecr.val[2], neon_p));

#define vec_by_scalar_mla_fpred_double_u16_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));\
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));

#define vec_by_scalar_mla_fpred_single_u16_x3(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));\
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vaddq_u16(vshrq_n_u16(vecr.val[2], 7), vandq_u16(vecr.val[2], neon_p));

#define vec_by_scalar_mla_fpred_single_u16_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u16(vshrq_n_u16(vecr.val[0], 7), vandq_u16(vecr.val[0], neon_p));\
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u16(vshrq_n_u16(vecr.val[1], 7), vandq_u16(vecr.val[1], neon_p));

#define vec_by_scalar_mla_u16_x3(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u16_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_fpred_mersenne_u32_x4(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u32(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u32(vandq_u32(vecr.val[0], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[0], 9), 3));\
    vecr.val[1] = vmlaq_laneq_u32(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u32(vandq_u32(vecr.val[1], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[1], 9), 3));\
    vecr.val[2] = vmlaq_laneq_u32(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vaddq_u32(vandq_u32(vecr.val[2], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[2], 9), 3));\
    vecr.val[3] = vmlaq_laneq_u32(vecr.val[3], veca.val[3], elemb.val[elem_i], elem_j); \
    vecr.val[3] = vaddq_u32(vandq_u32(vecr.val[3], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[3], 9), 3));

#define vec_by_scalar_mla_fpred_mersenne_u32_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u32(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[0] = vaddq_u32(vandq_u32(vecr.val[0], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[0], 9), 3));\
    vecr.val[1] = vmlaq_laneq_u32(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vaddq_u32(vandq_u32(vecr.val[1], neon_p), vmulq_n_u32(vshrq_n_u32(vecr.val[1], 9), 3));\

#define vec_by_scalar_mla_fpred_mersenne_u32_x1(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr = vmlaq_laneq_u32(vecr, veca, elemb.val[elem_i], elem_j); \
    vecr = vaddq_u32(vandq_u32(vecr, neon_p), vmulq_n_u32(vshrq_n_u32(vecr, 9), 3));

#define vec_by_scalar_mla_u32_x4(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u32(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u32(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vmlaq_laneq_u32(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    vecr.val[3] = vmlaq_laneq_u32(vecr.val[3], veca.val[3], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u32_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u32(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u32(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u32_x1(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr = vmlaq_laneq_u32(vecr, veca, elemb.val[elem_i], elem_j);

#if defined(HIGH_PERFORMANCE_AARCH64)
/* in-place normalization of redundant zero representation for syndromes*/
static inline
void fp_dz_norm_synd(FP_ELEM v[N-K]){
#if defined(RSDP)
#if defined(CATEGORY_1)
    uint8x16x4_t vv;
    uint8x16_t neon_one, neon_p;
    neon_one = vdupq_n_u8(1);
    neon_p = vdupq_n_u8(0x7f);
    vv = vld1q_u8_x4(v);
    for (int j = 0; j < 4; j++) {
        vv.val[j] = vandq_u8(vaddq_u8(vv.val[j],vshrq_n_u8(vaddq_u8(vv.val[j], neon_one),7)), neon_p);
    }
    vst1q_u8_x4(v, vv);
#elif defined(CATEGORY_3)
    uint8x16x4_t vv;
    uint8x16_t neon_one, neon_p;
    uint8x16_t vv3;
    neon_one = vdupq_n_u8(1);
    neon_p = vdupq_n_u8(0x7f);
    vv = vld1q_u8_x4(v);
    vv3 = vld1q_u8(v+64);
    for (int j = 0; j < 4; j++) {
        vv.val[j] = vandq_u8(vaddq_u8(vv.val[j],vshrq_n_u8(vaddq_u8(vv.val[j], neon_one),7)), neon_p);
    }
    vst1q_u8_x4(v, vv);
    vv3 = vandq_u8(vaddq_u8(vv3,vshrq_n_u8(vaddq_u8(vv3, neon_one),7)), neon_p);
    vst1q_u8(v+64, vv3);
#elif defined(CATEGORY_5)
    uint8x16x3_t vv;
    uint8x16_t neon_one, neon_p;
    neon_one = vdupq_n_u8(1);
    neon_p = vdupq_n_u8(0x7f);
    for (int i = 0; i < 96; i += 48) {
        vv = vld1q_u8_x3(v + i * sizeof(FP_ELEM));
        for (int j = 0; j < 3; j++) {
            vv.val[j] = vandq_u8(vaddq_u8(vv.val[j],vshrq_n_u8(vaddq_u8(vv.val[j], neon_one),7)), neon_p);
        }
        vst1q_u8_x3(v + i * sizeof(FP_ELEM), vv);
    }
    for (int i = 96; i < N-K; i++){
        v[i] = FP_DOUBLE_ZERO_NORM(v[i]);
    }
#endif
#elif defined(RSDPG)
    for (int i = 0; i < N-K; i++){
        v[i] = FP_DOUBLE_ZERO_NORM(v[i]);
    }
#endif
}
#else
static inline
void fp_dz_norm_synd(FP_ELEM v[N-K]){
    for (int i = 0; i < N-K; i++){
        v[i] = FP_DOUBLE_ZERO_NORM(v[i]);
    }
}
#endif

#if defined(HIGH_PERFORMANCE_AARCH64)
static inline
void fp_dz_norm(FP_ELEM v[N]){
#if defined(RSDP)
    uint8x16x4_t vv;
    uint8x16_t neon_one, neon_p;
    neon_one = vdupq_n_u8(1);
    neon_p = vdupq_n_u8(0x7f);
    for (int i = 0; i < N; i += 64) {
        vv = vld1q_u8_x4(v + i);
        for (int j = 0; j < 4; j++) {
            vv.val[j] = vandq_u8(vaddq_u8(vv.val[j],vshrq_n_u8(vaddq_u8(vv.val[j], neon_one),7)), neon_p);
        }
        vst1q_u8_x4(v + i, vv);
    }
#elif defined(RSDPG)
    for (int i = 0; i < N; i++){
       v[i] = FP_DOUBLE_ZERO_NORM(v[i]);
    }
#endif
}
#else
static inline
void fp_dz_norm(FP_ELEM v[N]){
    for (int i = 0; i < N; i++){
        v[i] = FP_DOUBLE_ZERO_NORM(v[i]);
    }
}
#endif

/* Computes the product e*H of an n-element restricted vector by a (n-k)*n
 * FP H is in systematic form. Only the non systematic portion of H =[V I],
 * V, is provided, transposed, hence linearized by columns so that syndrome
 * computation is vectorizable. */
#if (defined(HIGH_PERFORMANCE_AARCH64) && defined(RSDP))
static
void restr_vec_by_fp_matrix(FP_ELEM res[N-K],
                            FZ_ELEM e[N],
                            FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K,EPI16_PER_REG)]){
#if defined(CATEGORY_1)
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = (FP_DOUBLEPREC)RESTR_TO_VAL(e[i]);
    }
    uint16x8x4_t vr0;
    uint16x8x3_t vr1;
    vr0 = vld1q_u16_x4(restr_e + K);
    vr1 = vld1q_u16_x3(restr_e + K+32);
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8x3_t tmpV_tr1;
#pragma unroll
    for (int k = 0; k < 2; k++) { // k=0,1
        tmpe = vld1q_u16_x4(restr_e + k*32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(restr_e + 64);
    tmpe.val[1] = vld1q_u16(restr_e + 64 + 8);
    {
        tmpV_tr0 = vld1q_u16_x4(V_tr[64]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[64] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[65]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[65] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[66]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[66] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[67]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[67] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);

        tmpV_tr0 = vld1q_u16_x4(V_tr[68]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[68] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 4);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 4);

        tmpV_tr0 = vld1q_u16_x4(V_tr[69]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[69] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 5);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 5);

        tmpV_tr0 = vld1q_u16_x4(V_tr[70]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[70] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 6);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 6);

        tmpV_tr0 = vld1q_u16_x4(V_tr[71]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[71] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 7);
        vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 7);

        tmpV_tr0 = vld1q_u16_x4(V_tr[72]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[72] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 0);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[73]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[73] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 1);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[74]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[74] + 32);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 2);
        vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[75]);
        tmpV_tr1 = vld1q_u16_x3(V_tr[75] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 3);
        vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 3);
    }
    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
    }
    for (int i = 0; i < 3; i++) {
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
    }
    uint8x16x3_t vres0;
    uint8x8_t vres1;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres1 = vmovn_u16(vr1.val[2]);
    vst1q_u8_x3(res, vres0);
    vst1_u8(res + 48, vres1);
#elif  defined(CATEGORY_3)
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = (FP_DOUBLEPREC)RESTR_TO_VAL(e[i]);
    }
    uint16x8x4_t vr0, vr1;
    uint16x8x2_t vr2;
    vr0 = vld1q_u16_x4(restr_e + K);
    vr1 = vld1q_u16_x4(restr_e + (K+32));
    vr2 = vld1q_u16_x2(restr_e + (K+64));
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8x2_t tmpV_tr2;

#pragma unroll
    for (int k = 0; k < 3; k++) { // k=0,1,2
        tmpe = vld1q_u16_x4(restr_e + k * 32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+0] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+1] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+2] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+3] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+4] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+5] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+6] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(restr_e + 96);
    tmpe.val[1] = vld1q_u16(restr_e + 96 + 8);
    for (int i = 0; i < 2; i++) { // i=0,1
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+0]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+0] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+1]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+1] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+2]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+2] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+3]);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+3] + 64);
        vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+4]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+4] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 4);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+5]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+5] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 5);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+6]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+6] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+6] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 6);
        if (i == 0) {
            tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 7);
        }
    }
    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
    }
    for (int i = 0; i < 2; i++) {
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
    }
    uint8x16x4_t vres0;
    uint8x16_t vres1;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres0.val[3] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[2]), vreinterpretq_u8_u16(vr1.val[3]));
    vst1q_u8_x4(res, vres0);
    vres1 = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[0]), vreinterpretq_u8_u16(vr2.val[1]));
    vst1q_u8(res + 64, vres1);
#else
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = (FP_DOUBLEPREC)RESTR_TO_VAL(e[i]);
    }
    uint16x8x4_t vr0, vr1, vr2;
    uint16x8_t vr3;
    vr0 = vld1q_u16_x4(restr_e + K);
    vr1 = vld1q_u16_x4(restr_e + (K+32));
    vr2 = vld1q_u16_x4(restr_e + (K+64));
    vr3 = vld1q_u16(restr_e + (K+96));
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8_t tmpV_tr1;

#pragma unroll
    for (int k = 0; k < 4; k++) { // k=0,1,2,3
        tmpe = vld1q_u16_x4(restr_e + k * 32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+0] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+1] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+2] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+3] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+4] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+5] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+6] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+7] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(restr_e + 128);
    tmpe.val[1] = vld1q_u16(restr_e + 128 + 8);
    tmpe.val[2] = vld1q_u16(restr_e + 128 + 16);
    for (int i = 0; i < 3; i++) { // i=0,1,2
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+0] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+1] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+2] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3]);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3] + 64);
        vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+3] + 96);
        vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 3);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+4] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 4);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+5] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 5);

        if (i < 2) {
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr1 = vld1q_u16(V_tr[128+8*i+6] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr1 = vld1q_u16(V_tr[128+8*i+7] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }
    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
    }
    vr3 = vaddq_u16(vshrq_n_u16(vr3, 7), vandq_u16(vr3, neon_p));
    vr3 = vaddq_u16(vshrq_n_u16(vr3, 7), vandq_u16(vr3, neon_p));

    uint8x16x4_t vres0;
    uint8x16x2_t vres1;
    uint8x8_t vres2;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres0.val[3] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[2]), vreinterpretq_u8_u16(vr1.val[3]));
    vst1q_u8_x4(res, vres0);
    vres1.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[0]), vreinterpretq_u8_u16(vr2.val[1]));
    vres1.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[2]), vreinterpretq_u8_u16(vr2.val[3]));
    vst1q_u8_x2(res + 64, vres1);
    vres2 = vmovn_u16(vr3);
    vst1_u8(res + 96, vres2);
#endif
}
#elif defined(HIGH_PERFORMANCE_AARCH64) && defined(RSDPG)
static
void restr_vec_by_fp_matrix(FP_ELEM res[N-K],
                            FZ_ELEM e[N],
                            FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K, EPI32_PER_REG)]){
#if defined(CATEGORY_1)
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = TABLE_RESTR_TO_VAL[e[i]];
    }
    uint32x4x4_t vr0;
    uint32x4_t vr1;
    vr0 = vld1q_u32_x4(restr_e + K);
    vr1 = vld1q_u32(restr_e + (K+16));
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);
    uint32x4x4_t tmpe, tmpV_tr0;
    uint32x4_t tmpV_tr1;

#pragma unroll
    for (int k = 0; k < 2; k++) { // k=0,1
        tmpe = vld1q_u32_x4(restr_e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+0] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+1] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+2] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+3] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 3);
        }
    }

    tmpe.val[0] = vld1q_u32(restr_e + 32);
    {
        tmpV_tr0 = vld1q_u32_x4(V_tr[32]);
        tmpV_tr1 = vld1q_u32(V_tr[32] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);

        tmpV_tr0 = vld1q_u32_x4(V_tr[33]);
        tmpV_tr1 = vld1q_u32(V_tr[33] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);

        tmpV_tr0 = vld1q_u32_x4(V_tr[34]);
        tmpV_tr1 = vld1q_u32(V_tr[34] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);

        tmpV_tr0 = vld1q_u32_x4(V_tr[35]);
        tmpV_tr1 = vld1q_u32(V_tr[35] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);
    }

    uint16x8_t neon_v = vdupq_n_u16(16481);
    uint16x8_t neon_q = vdupq_n_u16(509);
    uint16x8_t vec2;
    uint32x4_t tmp, tmp2;
    uint16x8_t vres;
    uint16x4_t vres2;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    tmp = vmull_u16(vget_low_u16(vres), vget_low_u16(neon_v));
    tmp2 = vmull_high_u16(vres, neon_v);
    vec2 = vuzp2q_u16(vreinterpretq_u16_u32(tmp), vreinterpretq_u16_u32(tmp2));
    vec2 = vshrq_n_u16(vec2, 7);
    vres = vmlsq_u16(vres, vec2, neon_q);
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    tmp = vmull_u16(vget_low_u16(vres), vget_low_u16(neon_v));
    tmp2 = vmull_high_u16(vres, neon_v);
    vec2 = vuzp2q_u16(vreinterpretq_u16_u32(tmp), vreinterpretq_u16_u32(tmp2));
    vec2 = vshrq_n_u16(vec2, 7);
    vres = vmlsq_u16(vres, vec2, neon_q);
    vst1q_u16(res+8, vres);

    vr1 = vmlaq_n_u32(vandq_u32(vr1, neon_p), vshrq_n_u32(vr1, 9), 3);
    vres2 = vmovn_u32(vr1);
    uint16x4_t tmp3;
    tmp = vmull_n_u16(vres2, 16481);
    tmp = vshrq_n_u32(tmp, 23);
    tmp3 = vmovn_u32(tmp);
    vres2 = vmls_n_u16(vres2, tmp3, 509);
    vst1_u16(res+16, vres2);
#elif defined(CATEGORY_3)
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = TABLE_RESTR_TO_VAL[e[i]];
    }
    uint32x4x4_t vr0, vr2;
    vr0 = vld1q_u32_x4(restr_e + K);
    vr2 = vld1q_u32_x4(restr_e + (K+16));
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);
    uint32x4x4_t tmpe, tmpV_tr0, tmpV_tr2;

#pragma unroll
    for (int k = 0; k < 3; k++) { // k=0,1,2
        tmpe = vld1q_u32_x4(restr_e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+0] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+1] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+2] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+3] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 3);
        }
    }
    int16x8_t neon_v = vdupq_n_s16(16481);
    uint16x8_t neon_q = vdupq_n_u16(509);
    uint16x8_t tmp;
    int16x8_t tmp3;
    uint16x8_t vres;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+8, vres);

    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[0]), vreinterpretq_u16_u32(vr2.val[1]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+16, vres);

    vr2.val[2] = vmlaq_n_u32(vandq_u32(vr2.val[2], neon_p), vshrq_n_u32(vr2.val[2], 9), 3);
    vr2.val[2] = vmlaq_n_u32(vandq_u32(vr2.val[2], neon_p), vshrq_n_u32(vr2.val[2], 9), 3);
    vr2.val[3] = vmlaq_n_u32(vandq_u32(vr2.val[3], neon_p), vshrq_n_u32(vr2.val[3], 9), 3);
    vr2.val[3] = vmlaq_n_u32(vandq_u32(vr2.val[3], neon_p), vshrq_n_u32(vr2.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[2]), vreinterpretq_u16_u32(vr2.val[3]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+24, vres);
#elif defined(CATEGORY_5)
    FP_DOUBLEPREC restr_e[N];
    for (int i = 0 ;i < N; i++){
        restr_e[i] = TABLE_RESTR_TO_VAL[e[i]];
    }
    uint32x4x4_t vr0, vr1;
    uint32x4x2_t vr2;
    vr0 = vld1q_u32_x4(restr_e + K);
    vr1 = vld1q_u32_x4(restr_e + (K+16));
    vr2 = vld1q_u32_x2(restr_e + (K+32));
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);
    uint32x4x4_t tmpe, tmpV_tr0, tmpV_tr1;
    uint32x4x2_t tmpV_tr2;

#pragma unroll
    for (int k = 0; k < 4; k++) { // k=0,1,2,3
        tmpe = vld1q_u32_x4(restr_e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+0] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+0] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+1] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+1] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+2] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+2] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+3] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+3] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);
        }
    }

    tmpe.val[0] = vld1q_u32(restr_e + 64);
    {
        tmpV_tr0 = vld1q_u32_x4(V_tr[64]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[64] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[64] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 0);

        tmpV_tr0 = vld1q_u32_x4(V_tr[65]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[65] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[65] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 1);

        tmpV_tr0 = vld1q_u32_x4(V_tr[66]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[66] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[66] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 2);

        tmpV_tr0 = vld1q_u32_x4(V_tr[67]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[67] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[67] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 3);
    }

    uint16x8_t vres;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    vst1q_u16(res+8, vres);

    vr1.val[0] = vmlaq_n_u32(vandq_u32(vr1.val[0], neon_p), vshrq_n_u32(vr1.val[0], 9), 3);
    vr1.val[1] = vmlaq_n_u32(vandq_u32(vr1.val[1], neon_p), vshrq_n_u32(vr1.val[1], 9), 3);
    vr1.val[0] = vmlaq_n_u32(vandq_u32(vr1.val[0], neon_p), vshrq_n_u32(vr1.val[0], 9), 3);
    vr1.val[1] = vmlaq_n_u32(vandq_u32(vr1.val[1], neon_p), vshrq_n_u32(vr1.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr1.val[0]), vreinterpretq_u16_u32(vr1.val[1]));
    vst1q_u16(res+16, vres);

    vr1.val[2] = vmlaq_n_u32(vandq_u32(vr1.val[2], neon_p), vshrq_n_u32(vr1.val[2], 9), 3);
    vr1.val[3] = vmlaq_n_u32(vandq_u32(vr1.val[3], neon_p), vshrq_n_u32(vr1.val[3], 9), 3);
    vr1.val[2] = vmlaq_n_u32(vandq_u32(vr1.val[2], neon_p), vshrq_n_u32(vr1.val[2], 9), 3);
    vr1.val[3] = vmlaq_n_u32(vandq_u32(vr1.val[3], neon_p), vshrq_n_u32(vr1.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr1.val[2]), vreinterpretq_u16_u32(vr1.val[3]));
    vst1q_u16(res+24, vres);

    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[0]), vreinterpretq_u16_u32(vr2.val[1]));
    vst1q_u16(res+32, vres);

    for(int j = 0; j < N-K; j++){
        res[j] = FPRED_SINGLE( (FP_DOUBLEPREC) res[j] + restr_e[K-1]* V_tr[K-1][j]);
    }
#endif
}
#else
static
void restr_vec_by_fp_matrix(FP_ELEM res[N-K],
                            FZ_ELEM e[N],
                            FP_ELEM V_tr[K][N-K]){
    for (int i = K ;i < N; i++){
       res[i-K] = RESTR_TO_VAL(e[i]);
    }
    for(int i = 0; i < K; i++){
       for(int j = 0; j < N-K; j++){
           res[j] = FPRED_DOUBLE( (FP_DOUBLEPREC) res[j] +
                                  (FP_DOUBLEPREC) RESTR_TO_VAL(e[i]) *
                                  (FP_DOUBLEPREC) V_tr[i][j]);
       }
    }
}
#endif


#if (defined(HIGH_PERFORMANCE_AARCH64) && defined(RSDP))
static
void fp_vec_by_fp_matrix(FP_ELEM res[N-K],
                         FP_DOUBLEPREC e[N],
                         FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K,EPI16_PER_REG)]) {
#if defined(CATEGORY_1)
    uint16x8x4_t vr0;
    uint16x8x3_t vr1;
    vr0 = vld1q_u16_x4(e + K);
    vr1 = vld1q_u16_x3(e + K+32);
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8x3_t tmpV_tr1;

#pragma unroll
    for (int k = 0; k < 2; k++) { // k=0,1
        tmpe = vld1q_u16_x4(e + k*32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            tmpV_tr1 = vld1q_u16_x3(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(e + 64);
    tmpe.val[1] = vld1q_u16(e + 64 + 8);

    tmpV_tr0 = vld1q_u16_x4(V_tr[64]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[64] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);

    tmpV_tr0 = vld1q_u16_x4(V_tr[65]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[65] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);

    tmpV_tr0 = vld1q_u16_x4(V_tr[66]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[66] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);

    tmpV_tr0 = vld1q_u16_x4(V_tr[67]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[67] + 32);
    vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
    vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);

    tmpV_tr0 = vld1q_u16_x4(V_tr[68]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[68] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 4);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 4);

    tmpV_tr0 = vld1q_u16_x4(V_tr[69]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[69] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 5);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 5);

    tmpV_tr0 = vld1q_u16_x4(V_tr[70]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[70] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 6);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 6);

    tmpV_tr0 = vld1q_u16_x4(V_tr[71]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[71] + 32);
    vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 7);
    vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 0, 7);

    tmpV_tr0 = vld1q_u16_x4(V_tr[72]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[72] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 0);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 0);

    tmpV_tr0 = vld1q_u16_x4(V_tr[73]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[73] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 1);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 1);

    tmpV_tr0 = vld1q_u16_x4(V_tr[74]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[74] + 32);
    vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 2);
    vec_by_scalar_mla_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 2);

    tmpV_tr0 = vld1q_u16_x4(V_tr[75]);
    tmpV_tr1 = vld1q_u16_x3(V_tr[75] + 32);
    vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, 1, 3);
    vec_by_scalar_mla_fpred_single_u16_x3(vr1, tmpV_tr1, tmpe, neon_p, 1, 3);

    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
    }
    for (int i = 0; i < 3; i++) {
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
    }

    uint8x16x3_t vres0;
    uint8x8_t vres1;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres1 = vmovn_u16(vr1.val[2]);
    vst1q_u8_x3(res, vres0);
    vst1_u8(res + 48, vres1);
#elif defined(CATEGORY_3)
    uint16x8x4_t vr0, vr1;
    uint16x8x2_t vr2;
    vr0 = vld1q_u16_x4(e + K);
    vr1 = vld1q_u16_x4(e + (K+32));
    vr2 = vld1q_u16_x2(e + (K+64));
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8x2_t tmpV_tr2;

#pragma unroll
    for (int k = 0; k < 3; k++) { // k=0,1,2
        tmpe = vld1q_u16_x4(e + k * 32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+0] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+1] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+2] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+3] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+4] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+5] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+6] + 64);
            vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr2 = vld1q_u16_x2(V_tr[32*k+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(e + 96);
    tmpe.val[1] = vld1q_u16(e + 96 + 8);
    for (int i = 0; i < 2; i++) { // i=0,1
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+0]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+0] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+1]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+1] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+2]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+2] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+3]);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+3] + 64);
        vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+4]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+4] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 4);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+5]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+5] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 5);

        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+6]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
        tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+6] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
        tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+6] + 64);
        vec_by_scalar_mla_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 6);
        if (i == 0) {
            tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[96+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr2 = vld1q_u16_x2(V_tr[96+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 7);
        }
    }
    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
    }
    for (int i = 0; i < 2; i++) {
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
    }

    uint8x16x4_t vres0;
    uint8x16_t vres1;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres0.val[3] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[2]), vreinterpretq_u8_u16(vr1.val[3]));
    vst1q_u8_x4(res, vres0);
    vres1 = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[0]), vreinterpretq_u8_u16(vr2.val[1]));
    vst1q_u8(res + 64, vres1);
#elif defined(CATEGORY_5)
    uint16x8x4_t vr0, vr1, vr2;
    uint16x8_t vr3;
    vr0 = vld1q_u16_x4(e + K);
    vr1 = vld1q_u16_x4(e + (K+32));
    vr2 = vld1q_u16_x4(e + (K+64));
    vr3 = vld1q_u16(e + (K+96));
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint16x8x4_t tmpe, tmpV_tr0;
    uint16x8_t tmpV_tr1;

#pragma unroll
    for (int k = 0; k < 4; k++) { // k=0,1,2,3
        tmpe = vld1q_u16_x4(e + k * 32);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+0] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 0);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+0] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+1] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 1);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+1] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+2] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 2);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+2] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+3] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 3);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+3] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 3);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+4] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 4);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+4] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 4);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+5] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 5);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+5] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 5);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+6] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+6] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[32*k+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr1 = vld1q_u16(V_tr[32*k+8*i+7] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }

    tmpe.val[0] = vld1q_u16(e + 128);
    tmpe.val[1] = vld1q_u16(e + 128 + 8);
    tmpe.val[2] = vld1q_u16(e + 128 + 16);
    for (int i = 0; i < 3; i++) { // i=0,1,2
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+0] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 0);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+0] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 0);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+1] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 1);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+1] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 1);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+2] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 2);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+2] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 2);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3]);
        vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+3] + 64);
        vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 3);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+3] + 96);
        vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 3);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+4] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 4);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+4] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 4);

        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5]);
        vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+5] + 64);
        vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 5);
        tmpV_tr1 = vld1q_u16(V_tr[128+8*i+5] + 96);
        vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 5);

        if (i < 2) {
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6]);
            vec_by_scalar_mla_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6] + 32);
            vec_by_scalar_mla_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+6] + 64);
            vec_by_scalar_mla_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 6);
            tmpV_tr1 = vld1q_u16(V_tr[128+8*i+6] + 96);
            vec_by_scalar_mla_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 6);

            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7]);
            vec_by_scalar_mla_fpred_single_u16_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7] + 32);
            vec_by_scalar_mla_fpred_single_u16_x4(vr1, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr0 = vld1q_u16_x4(V_tr[128+8*i+7] + 64);
            vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpV_tr0, tmpe, neon_p, i, 7);
            tmpV_tr1 = vld1q_u16(V_tr[128+8*i+7] + 96);
            vec_by_scalar_mla_fpred_single_u16_x1(vr3, tmpV_tr1, tmpe, neon_p, i, 7);
        }
    }
    for (int i = 0; i < 4; i++) {
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr0.val[i] = vaddq_u16(vshrq_n_u16(vr0.val[i], 7), vandq_u16(vr0.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr1.val[i] = vaddq_u16(vshrq_n_u16(vr1.val[i], 7), vandq_u16(vr1.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
    }
    vr3 = vaddq_u16(vshrq_n_u16(vr3, 7), vandq_u16(vr3, neon_p));
    vr3 = vaddq_u16(vshrq_n_u16(vr3, 7), vandq_u16(vr3, neon_p));

    uint8x16x4_t vres0;
    uint8x16x2_t vres1;
    uint8x8_t vres2;
    vres0.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[0]), vreinterpretq_u8_u16(vr0.val[1]));
    vres0.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr0.val[2]), vreinterpretq_u8_u16(vr0.val[3]));
    vres0.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[0]), vreinterpretq_u8_u16(vr1.val[1]));
    vres0.val[3] = vuzp1q_u8(vreinterpretq_u8_u16(vr1.val[2]), vreinterpretq_u8_u16(vr1.val[3]));
    vst1q_u8_x4(res, vres0);
    vres1.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[0]), vreinterpretq_u8_u16(vr2.val[1]));
    vres1.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[2]), vreinterpretq_u8_u16(vr2.val[3]));
    vst1q_u8_x2(res + 64, vres1);
    vres2 = vmovn_u16(vr3);
    vst1_u8(res + 96, vres2);
#endif
}
#elif (defined(HIGH_PERFORMANCE_AARCH64) && defined(RSDPG))
static
void fp_vec_by_fp_matrix(FP_ELEM res[N-K],
                         FP_DOUBLEPREC e[ROUND_UP(N,EPI32_PER_REG)],
                         FP_DOUBLEPREC V_tr[K][ROUND_UP(N-K,EPI32_PER_REG)]) {
#if defined(CATEGORY_1)
    uint32x4x4_t vr0;
    uint32x4_t vr1;
    vr0 = vld1q_u32_x4(e + K);
    vr1 = vld1q_u32(e + (K+16));
    uint32x4x4_t tmpe, tmpV_tr0;
    uint32x4_t tmpV_tr1;
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);

#pragma unroll
    for (int k = 0; k < 2; k++) { // k=0,1
        tmpe = vld1q_u32_x4(e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+0] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+1] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+2] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr1 = vld1q_u32(V_tr[16*k+4*i+3] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, i, 3);
        }
    }

    tmpe.val[0] = vld1q_u32(e + 32);
    {
        tmpV_tr0 = vld1q_u32_x4(V_tr[32]);
        tmpV_tr1 = vld1q_u32(V_tr[32] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);

        tmpV_tr0 = vld1q_u32_x4(V_tr[33]);
        tmpV_tr1 = vld1q_u32(V_tr[33] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);

        tmpV_tr0 = vld1q_u32_x4(V_tr[34]);
        tmpV_tr1 = vld1q_u32(V_tr[34] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);

        tmpV_tr0 = vld1q_u32_x4(V_tr[35]);
        tmpV_tr1 = vld1q_u32(V_tr[35] + 16);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x1(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);
    }

    uint16x8_t neon_v = vdupq_n_u16(16481);
    uint16x8_t neon_q = vdupq_n_u16(509);
    uint16x8_t vec2;
    uint32x4_t tmp, tmp2;
    uint16x8_t vres;
    uint16x4_t vres2;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    tmp = vmull_u16(vget_low_u16(vres), vget_low_u16(neon_v));
    tmp2 = vmull_high_u16(vres, neon_v);
    vec2 = vuzp2q_u16(vreinterpretq_u16_u32(tmp), vreinterpretq_u16_u32(tmp2));
    vec2 = vshrq_n_u16(vec2, 7);
    vres = vmlsq_u16(vres, vec2, neon_q);
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    tmp = vmull_u16(vget_low_u16(vres), vget_low_u16(neon_v));
    tmp2 = vmull_high_u16(vres, neon_v);
    vec2 = vuzp2q_u16(vreinterpretq_u16_u32(tmp), vreinterpretq_u16_u32(tmp2));
    vec2 = vshrq_n_u16(vec2, 7);
    vres = vmlsq_u16(vres, vec2, neon_q);
    vst1q_u16(res+8, vres);

    vr1 = vmlaq_n_u32(vandq_u32(vr1, neon_p), vshrq_n_u32(vr1, 9), 3);
    vres2 = vmovn_u32(vr1);
    uint16x4_t tmp3;
    tmp = vmull_n_u16(vres2, 16481);
    tmp = vshrq_n_u32(tmp, 23);
    tmp3 = vmovn_u32(tmp);
    vres2 = vmls_n_u16(vres2, tmp3, 509);
    vst1_u16(res+16, vres2);
#elif defined(CATEGORY_3)
    uint32x4x4_t vr0, vr2;
    vr0 = vld1q_u32_x4(e + K);
    vr2 = vld1q_u32_x4(e + (K+16));
    uint32x4x4_t tmpe, tmpV_tr0, tmpV_tr2;
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);

#pragma unroll
    for (int k = 0; k < 3; k++) { // k=0,1,2
        tmpe = vld1q_u32_x4(e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+0] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+1] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+2] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr2 = vld1q_u32_x4(V_tr[16*k+4*i+3] + 16);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x4(vr2, tmpV_tr2, tmpe, neon_p, i, 3);
        }
    }

    int16x8_t neon_v = vdupq_n_s16(16481);
    uint16x8_t neon_q = vdupq_n_u16(509);
    uint16x8_t tmp;
    int16x8_t tmp3;
    uint16x8_t vres;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+8, vres);

    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[0]), vreinterpretq_u16_u32(vr2.val[1]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+16, vres);

    vr2.val[2] = vmlaq_n_u32(vandq_u32(vr2.val[2], neon_p), vshrq_n_u32(vr2.val[2], 9), 3);
    vr2.val[2] = vmlaq_n_u32(vandq_u32(vr2.val[2], neon_p), vshrq_n_u32(vr2.val[2], 9), 3);
    vr2.val[3] = vmlaq_n_u32(vandq_u32(vr2.val[3], neon_p), vshrq_n_u32(vr2.val[3], 9), 3);
    vr2.val[3] = vmlaq_n_u32(vandq_u32(vr2.val[3], neon_p), vshrq_n_u32(vr2.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[2]), vreinterpretq_u16_u32(vr2.val[3]));
    tmp3 = vqdmulhq_s16(vreinterpretq_s16_u16(vres), neon_v);
    tmp = vshrq_n_u16(vreinterpretq_u16_s16(tmp3), 8);
    vres = vmlsq_u16(vres, tmp, neon_q);
    vst1q_u16(res+24, vres);
#elif defined(CATEGORY_5)
    uint32x4x4_t vr0, vr1;
    uint32x4x2_t vr2;
    vr0 = vld1q_u32_x4(e + K);
    vr1 = vld1q_u32_x4(e + (K+16));
    vr2 = vld1q_u32_x2(e + (K+32));
    uint32x4x4_t tmpe, tmpV_tr0, tmpV_tr1;
    uint32x4x2_t tmpV_tr2;
    uint32x4_t neon_p = vdupq_n_u32(0x1ff);

#pragma unroll
    for (int k = 0; k < 4; k++) { // k=0,1,2,3
        tmpe = vld1q_u32_x4(e + k * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) { // i=0,1,2,3
            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+0]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+0] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+0] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 0);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 0);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+1]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+1] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+1] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 1);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 1);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+2]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+2] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+2] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 2);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 2);

            tmpV_tr0 = vld1q_u32_x4(V_tr[16*k+4*i+3]);
            tmpV_tr1 = vld1q_u32_x4(V_tr[16*k+4*i+3] + 16);
            tmpV_tr2 = vld1q_u32_x2(V_tr[16*k+4*i+3] + 32);
            vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, i, 3);
            vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, i, 3);
        }
    }

    tmpe.val[0] = vld1q_u32(e + 64);
    {
        tmpV_tr0 = vld1q_u32_x4(V_tr[64]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[64] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[64] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 0);

        tmpV_tr0 = vld1q_u32_x4(V_tr[65]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[65] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[65] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 1);

        tmpV_tr0 = vld1q_u32_x4(V_tr[66]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[66] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[66] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 2);

        tmpV_tr0 = vld1q_u32_x4(V_tr[67]);
        tmpV_tr1 = vld1q_u32_x4(V_tr[67] + 16);
        tmpV_tr2 = vld1q_u32_x2(V_tr[67] + 32);
        vec_by_scalar_mla_u32_x4(vr0, tmpV_tr0, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x4(vr1, tmpV_tr1, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_u32_x2(vr2, tmpV_tr2, tmpe, neon_p, 0, 3);
    }


    uint16x8_t vres;
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vr0.val[0] = vmlaq_n_u32(vandq_u32(vr0.val[0], neon_p), vshrq_n_u32(vr0.val[0], 9), 3);
    vr0.val[1] = vmlaq_n_u32(vandq_u32(vr0.val[1], neon_p), vshrq_n_u32(vr0.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[0]), vreinterpretq_u16_u32(vr0.val[1]));
    vst1q_u16(res, vres);

    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vr0.val[2] = vmlaq_n_u32(vandq_u32(vr0.val[2], neon_p), vshrq_n_u32(vr0.val[2], 9), 3);
    vr0.val[3] = vmlaq_n_u32(vandq_u32(vr0.val[3], neon_p), vshrq_n_u32(vr0.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr0.val[2]), vreinterpretq_u16_u32(vr0.val[3]));
    vst1q_u16(res+8, vres);

    vr1.val[0] = vmlaq_n_u32(vandq_u32(vr1.val[0], neon_p), vshrq_n_u32(vr1.val[0], 9), 3);
    vr1.val[1] = vmlaq_n_u32(vandq_u32(vr1.val[1], neon_p), vshrq_n_u32(vr1.val[1], 9), 3);
    vr1.val[0] = vmlaq_n_u32(vandq_u32(vr1.val[0], neon_p), vshrq_n_u32(vr1.val[0], 9), 3);
    vr1.val[1] = vmlaq_n_u32(vandq_u32(vr1.val[1], neon_p), vshrq_n_u32(vr1.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr1.val[0]), vreinterpretq_u16_u32(vr1.val[1]));
    vst1q_u16(res+16, vres);

    vr1.val[2] = vmlaq_n_u32(vandq_u32(vr1.val[2], neon_p), vshrq_n_u32(vr1.val[2], 9), 3);
    vr1.val[3] = vmlaq_n_u32(vandq_u32(vr1.val[3], neon_p), vshrq_n_u32(vr1.val[3], 9), 3);
    vr1.val[2] = vmlaq_n_u32(vandq_u32(vr1.val[2], neon_p), vshrq_n_u32(vr1.val[2], 9), 3);
    vr1.val[3] = vmlaq_n_u32(vandq_u32(vr1.val[3], neon_p), vshrq_n_u32(vr1.val[3], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr1.val[2]), vreinterpretq_u16_u32(vr1.val[3]));
    vst1q_u16(res+24, vres);

    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vr2.val[0] = vmlaq_n_u32(vandq_u32(vr2.val[0], neon_p), vshrq_n_u32(vr2.val[0], 9), 3);
    vr2.val[1] = vmlaq_n_u32(vandq_u32(vr2.val[1], neon_p), vshrq_n_u32(vr2.val[1], 9), 3);
    vres = vuzp1q_u16(vreinterpretq_u16_u32(vr2.val[0]), vreinterpretq_u16_u32(vr2.val[1]));
    vst1q_u16(res+32, vres);

    for(int j = 0; j < N-K; j++){
        res[j] = FPRED_SINGLE( (FP_DOUBLEPREC) res[j] + e[K-1] * V_tr[K-1][j]);
    }
#endif
}
#else
static
void fp_vec_by_fp_matrix(FP_ELEM res[N-K],
                         FP_ELEM e[N],
                         FP_ELEM V_tr[K][N-K]){
    memcpy(res,e+K,(N-K)*sizeof(FP_ELEM));
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N-K; j++){
            res[j] = FPRED_DOUBLE( (FP_DOUBLEPREC) res[j] +
                                   (FP_DOUBLEPREC) e[i] *
                                   (FP_DOUBLEPREC) V_tr[i][j]);
        }
    }
}
#endif


static inline
void fp_vec_by_fp_vec_pointwise(FP_ELEM res[N],
                                const FP_ELEM in1[N],
                                const FP_ELEM in2[N]){
    for(int i = 0; i < N; i++){
        res[i] = FPRED_DOUBLE( (FP_DOUBLEPREC) in1[i] *
                               (FP_DOUBLEPREC) in2[i] );
    }
}

static inline
void restr_by_fp_vec_pointwise(FP_ELEM res[N],
                                const FZ_ELEM in1[N],
                                const FP_ELEM in2[N]){
#if defined(RSDP)
    for(int i = 0; i < N; i++){
        res[i] = FPRED_DOUBLE( (FP_DOUBLEPREC) RESTR_TO_VAL(in1[i]) *
                               (FP_DOUBLEPREC) in2[i]);
    }
#else
    for(int i = 0; i < N; i++){
        res[i] = FPRED_DOUBLE( (FP_DOUBLEPREC) TABLE_RESTR_TO_VAL[in1[i]] *
                               (FP_DOUBLEPREC) in2[i]);
    }
#endif
}

/* e*chall_1 + u_prime*/
static inline
void fp_vec_by_restr_vec_scaled(FP_ELEM res[N],
                                const FZ_ELEM e[N],
                                const FP_ELEM chall_1,
                                const FP_ELEM u_prime[N]){
#if defined(RSDP)
    for(int i = 0; i < N; i++){
        res[i] = FPRED_DOUBLE( (FP_DOUBLEPREC) u_prime[i] +
                               (FP_DOUBLEPREC) RESTR_TO_VAL(e[i]) * (FP_DOUBLEPREC) chall_1) ;
    }
#else
    for(int i = 0; i < N; i++){
        res[i] = FPRED_DOUBLE( (FP_DOUBLEPREC) u_prime[i] +
                               (FP_DOUBLEPREC) TABLE_RESTR_TO_VAL[e[i]] * (FP_DOUBLEPREC) chall_1) ;
    }
#endif
}

static inline
void fp_synd_minus_fp_vec_scaled(FP_ELEM res[N-K],
                                 const FP_ELEM synd[N-K],
                                 const FP_ELEM chall_1,
                                 const FP_ELEM s[N-K]) {
    for(int j = 0; j < N-K; j++){
        FP_ELEM tmp = FPRED_DOUBLE( (FP_DOUBLEPREC) s[j] * (FP_DOUBLEPREC) chall_1);
        tmp = FP_DOUBLE_ZERO_NORM(tmp);
        res[j] = FPRED_SINGLE( (FP_DOUBLEPREC) synd[j] + FPRED_OPPOSITE(tmp) );
    }
}

static inline
void convert_restr_vec_to_fp(FP_ELEM res[N],
                            const FZ_ELEM in[N]){
#if defined(RSDP)
    for(int j = 0; j < N; j++){
        res[j] = RESTR_TO_VAL(in[j]);
    }
#else
    for(int j = 0; j < N; j++){
        res[j] = TABLE_RESTR_TO_VAL[in[j]];
    }
#endif
}
