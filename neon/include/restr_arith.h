/*
 * NeonCROSS: Vectorized Implementation of the Post-Quantum Signature Algorithm CROSS
 * Copyright (c) 2025 Hanyu Wei et al.
 * Licensed under the Apache License, Version 2.0; see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "architecture_detect.h"
#include "parameters.h"

#if defined(RSDP)
#define FZRED_SINGLE(x)   (((x) & 0x07) + ((x) >> 3))
#define FZRED_OPPOSITE(x) ((x) ^ 0x07)
#define FZ_DOUBLE_ZERO_NORM(x) (((x) + (((x) + 1) >> 3)) & 0x07)

#elif defined(RSDPG)
#define FZRED_SINGLE(x)   (((x) & 0x7f) + ((x) >> 7))
#define FZRED_DOUBLE(x) FZRED_SINGLE(FZRED_SINGLE(x))
#define FZRED_OPPOSITE(x) ((x) ^ 0x7f)
#define FZ_DOUBLE_ZERO_NORM(x) (((x) + (((x) + 1) >> 7)) & 0x7f)
#endif

static inline
void fz_dz_norm_n(FZ_ELEM v[N]){
    for (int i = 0; i < N; i++){
       v[i] = FZ_DOUBLE_ZERO_NORM(v[i]);
    }
}

/* Elements of the restricted subgroups are represented as the exponents of
 * the generator */
static inline
void fz_vec_sub_n(FZ_ELEM res[N],
                  const FZ_ELEM a[N],
                  const FZ_ELEM b[N]){
    for(int i = 0; i < N; i++){
        res[i]= FZRED_SINGLE( a[i] + FZRED_OPPOSITE(b[i]) );
    }
}

static inline
int is_fz_vec_in_restr_group_n(const FZ_ELEM in[N]){
    int is_in_ok = 1;
    for(int i=0; i<N; i++){
        is_in_ok = is_in_ok && (in[i] < Z);
    }
    return is_in_ok;
}

#define vmovl_u8_xidx(a, b, idx) \
    for(int i = 0; i < idx; i++) \
    a.val[i] = vmovl_u8(b.val[i]); \

#define vmovn_u16_xidx(a, b, idx) \
    for(int i = 0; i < idx; i++)  \
    a.val[i] = vmovn_u16(b.val[i]); \


#define pmp_reduction_u16_mod127(vec, neon_p) \
    vec = vaddq_u16(vshrq_n_u16(vec, 7), vandq_u16(vec, neon_p));

#define vec_by_scalar_mla_fpred_double_u16_x4(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[0], neon_p); \
    pmp_reduction_u16_mod127(vecr.val[0], neon_p); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[1], neon_p); \
    pmp_reduction_u16_mod127(vecr.val[1], neon_p); \
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[2], neon_p); \
    pmp_reduction_u16_mod127(vecr.val[2], neon_p); \
    vecr.val[3] = vmlaq_laneq_u16(vecr.val[3], veca.val[3], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[3], neon_p); \
    pmp_reduction_u16_mod127(vecr.val[3], neon_p); 

#define vec_by_scalar_mla_fpred_double_u16_x1(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr = vmlaq_laneq_u16(vecr, veca, elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr, neon_p); \
    pmp_reduction_u16_mod127(vecr, neon_p); 

#define vec_by_scalar_mla_fpred_single_u16_x4(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[0], neon_p); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[1], neon_p); \
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[2], neon_p); \
    vecr.val[3] = vmlaq_laneq_u16(vecr.val[3], veca.val[3], elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr.val[3], neon_p); 

#define vec_by_scalar_mla_fpred_single_u16_x1(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr = vmlaq_laneq_u16(vecr, veca, elemb.val[elem_i], elem_j); \
    pmp_reduction_u16_mod127(vecr, neon_p); 

#define vec_by_scalar_mla_u16_x4(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j); \
    vecr.val[3] = vmlaq_laneq_u16(vecr.val[3], veca.val[3], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u16_x3(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j); \
    vecr.val[2] = vmlaq_laneq_u16(vecr.val[2], veca.val[2], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u16_x2(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr.val[0] = vmlaq_laneq_u16(vecr.val[0], veca.val[0], elemb.val[elem_i], elem_j); \
    vecr.val[1] = vmlaq_laneq_u16(vecr.val[1], veca.val[1], elemb.val[elem_i], elem_j);

#define vec_by_scalar_mla_u16_x1(vecr, veca, elemb, neon_p, elem_i, elem_j) \
    vecr = vmlaq_laneq_u16(vecr, veca, elemb.val[elem_i], elem_j);

extern void fz_inf_w_by_fz_matrix_lv5(uint8_t *, uint16_t *, uint16_t *);
extern void fz_inf_w_by_fz_matrix_lv3(uint8_t *, uint16_t *, uint16_t *);
extern void fz_inf_w_by_fz_matrix_lv1(uint8_t *, uint16_t *, uint16_t *);

#if defined(RSDPG)
/* computes the information word * M_G product to obtain an element of G
 * only non systematic portion of M_G = [W I] is used, transposed to improve
 * cache friendliness */
#if defined(HIGH_PERFORMANCE_AARCH64)
static
void fz_inf_w_by_fz_matrix(FZ_ELEM res[N],
                           const FZ_ELEM e[M],
                           FZ_DOUBLEPREC W_mat[M][ROUND_UP(N-M, EPI16_PER_REG)]){
#if defined(CATEGORY_1)
#ifdef __APPLE__
    uint16x8x4_t vr;
    vr.val[0] = vdupq_n_u16(0);
    vr.val[1] = vdupq_n_u16(0);
    vr.val[2] = vdupq_n_u16(0);
    vr.val[3] = vdupq_n_u16(0);
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint8x8x3_t ve;
    uint16x8x3_t tmpe;
    uint16x8x4_t tmpW_mat;
    ve = vld1_u8_x3(e);
    vmovl_u8_xidx(tmpe, ve, 3);
    for(int i = 0; i < 3; i++) { // i = 0,1,2;  M=25=8*3+1
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+0]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 0);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+1]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 1);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+2]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 2);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+3]);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 3);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+4]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 4);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+5]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 5);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+6]);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 6);
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+7]);
        vec_by_scalar_mla_fpred_double_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 7);
    }
    for (int i = 0; i < 4;i++) {
        vr.val[i] = vaddq_u16(vshrq_n_u16(vr.val[i], 7), vandq_u16(vr.val[i], neon_p));
    }
    uint8x16x2_t vres;
    vres.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[0]), vreinterpretq_u8_u16(vr.val[1]));
    vres.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[2]), vreinterpretq_u8_u16(vr.val[3]));
    vst1q_u8_x2(res, vres);
    for(int j = 0; j < N-M; j++){
        res[j] = FZRED_DOUBLE( (FZ_DOUBLEPREC) res[j] +
                                  (FZ_DOUBLEPREC) e[M-1] *
                                  (FZ_DOUBLEPREC) W_mat[M-1][j]);
    }
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));
#else
    FZ_DOUBLEPREC e_neon[ROUND_UP(M,EPI16_PER_REG)] = {0};
    for(int i = 0; i < M; i++) {
        e_neon[i] = (FZ_DOUBLEPREC)e[i];
    }
    fz_inf_w_by_fz_matrix_lv1(res, e_neon, (uint16_t *)W_mat);
    for(int j = 0; j < N-M; j++){
        res[j] = FZRED_DOUBLE( (FZ_DOUBLEPREC) res[j] +
                                  (FZ_DOUBLEPREC) e[M-1] *
                                  (FZ_DOUBLEPREC) W_mat[M-1][j]);
    }
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));
#endif
#elif defined(CATEGORY_3)
#ifdef __APPLE__
    uint16x8x4_t vr;
    uint16x8_t vr1;
    vr.val[0] = vdupq_n_u16(0);
    vr.val[1] = vdupq_n_u16(0);
    vr.val[2] = vdupq_n_u16(0);
    vr.val[3] = vdupq_n_u16(0);
    vr1 = vdupq_n_u16(0);
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint8x8x4_t ve;
    uint16x8x4_t tmpe;
    uint16x8x4_t tmpW_mat;
    uint16x8_t tmpW_mat1;

    ve = vld1_u8_x4(e);
    vmovl_u8_xidx(tmpe, ve, 4);
    for(int i = 0; i < 4; i++) { // i = 0,1,2,3;  M=40=8*4+8
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+0]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 0);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 0);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+1]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 1);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 1);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+2]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 2);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 2);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+3]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 3);
        vec_by_scalar_mla_fpred_single_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 3);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+4]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 4);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 4);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+5]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 5);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 5);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+6]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+6] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 6);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 6);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+7]);
        tmpW_mat1 = vld1q_u16(W_mat[8*i+7] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 7);
        vec_by_scalar_mla_fpred_single_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, i, 7);
    }
    ve.val[0] = vld1_u8(e + 32);
    tmpe.val[0] = vmovl_u8(ve.val[0]);
    {
        tmpW_mat = vld1q_u16_x4(W_mat[32]);
        tmpW_mat1 = vld1q_u16(W_mat[32] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 0);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 0);

        tmpW_mat = vld1q_u16_x4(W_mat[33]);
        tmpW_mat1 = vld1q_u16(W_mat[33] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 1);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 1);

        tmpW_mat = vld1q_u16_x4(W_mat[34]);
        tmpW_mat1 = vld1q_u16(W_mat[34] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 2);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 2);

        tmpW_mat = vld1q_u16_x4(W_mat[35]);
        tmpW_mat1 = vld1q_u16(W_mat[35] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 3);
        vec_by_scalar_mla_fpred_single_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 3);

        tmpW_mat = vld1q_u16_x4(W_mat[36]);
        tmpW_mat1 = vld1q_u16(W_mat[36] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 4);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 4);

        tmpW_mat = vld1q_u16_x4(W_mat[37]);
        tmpW_mat1 = vld1q_u16(W_mat[37] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 5);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 5);

        tmpW_mat = vld1q_u16_x4(W_mat[38]);
        tmpW_mat1 = vld1q_u16(W_mat[38] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 6);
        vec_by_scalar_mla_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 6);

        tmpW_mat = vld1q_u16_x4(W_mat[39]);
        tmpW_mat1 = vld1q_u16(W_mat[39] + 32);
        vec_by_scalar_mla_fpred_double_u16_x4(vr, tmpW_mat, tmpe, neon_p, 0, 7);
        vec_by_scalar_mla_fpred_double_u16_x1(vr1, tmpW_mat1, tmpe, neon_p, 0, 7);
    }
    for (int i = 0; i < 4;i++) {
        vr.val[i] = vaddq_u16(vshrq_n_u16(vr.val[i], 7), vandq_u16(vr.val[i], neon_p));
    }
    vr1 = vaddq_u16(vshrq_n_u16(vr1, 7), vandq_u16(vr1, neon_p));

    uint8x16x2_t vres;
    uint8x8_t vres1;
    vres.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[0]), vreinterpretq_u8_u16(vr.val[1]));
    vres.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[2]), vreinterpretq_u8_u16(vr.val[3]));
    vst1q_u8_x2(res, vres);
    vres1 = vmovn_u16(vr1);
    vst1_u8(res+32, vres1);
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));
#else
    FZ_DOUBLEPREC e_neon[ROUND_UP(M,EPI16_PER_REG)] = {0};
    for(int i = 0; i < M; i++) {
        e_neon[i] = (FZ_DOUBLEPREC)e[i];
    }
    fz_inf_w_by_fz_matrix_lv3(res, e_neon, (uint16_t *)W_mat);
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));  
#endif
    
#elif defined(CATEGORY_5)
#ifdef __APPLE__
    uint16x8x4_t vr ,vr2;
    vr.val[0] = vdupq_n_u16(0);
    vr.val[1] = vdupq_n_u16(0);
    vr.val[2] = vdupq_n_u16(0);
    vr.val[3] = vdupq_n_u16(0);
    vr2.val[0] = vdupq_n_u16(0);
    vr2.val[1] = vdupq_n_u16(0);
    vr2.val[2] = vdupq_n_u16(0);
    vr2.val[3] = vdupq_n_u16(0);
    uint16x8_t neon_p;
    neon_p = vdupq_n_u16(0x7f);
    uint8x8x4_t ve;
    uint16x8x4_t tmpe;
    uint16x8x4_t tmpW_mat, tmpW_mat2;

    ve = vld1_u8_x4(e);
    vmovl_u8_xidx(tmpe, ve, 4);
    for(int i = 0; i < 4; i++) { // i = 0,1,2,3;  M=48=8*4+8*2
        tmpW_mat = vld1q_u16_x4(W_mat[8*i+0]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 0);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 0);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+1]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 1);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 1);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+2]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 2);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 2);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+3]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 3);
        vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 3);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+4]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 4);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 4);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+5]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 5);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 5);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+6]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+6] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 6);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 6);

        tmpW_mat = vld1q_u16_x4(W_mat[8*i+7]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[8*i+7] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 7);
        vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 7);
    }
    ve.val[0] = vld1_u8(e + 32);
    ve.val[1] = vld1_u8(e + 32 + 8);
    tmpe.val[0] = vmovl_u8(ve.val[0]);
    tmpe.val[1] = vmovl_u8(ve.val[1]);
    for(int i = 0; i < 2; i++) { // i = 0,1;  M=48=8*4+8*2
        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+0]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+0] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 0);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 0);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+1]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+1] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 1);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 1);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+2]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+2] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 2);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 2);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+3]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+3] + 32);
        vec_by_scalar_mla_fpred_single_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 3);
        vec_by_scalar_mla_fpred_single_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 3);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+4]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+4] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 4);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 4);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+5]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+5] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 5);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 5);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+6]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+6] + 32);
        vec_by_scalar_mla_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 6);
        vec_by_scalar_mla_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 6);

        tmpW_mat = vld1q_u16_x4(W_mat[32+8*i+7]);
        tmpW_mat2 = vld1q_u16_x4(W_mat[32+8*i+7] + 32);
        vec_by_scalar_mla_fpred_double_u16_x4(vr, tmpW_mat, tmpe, neon_p, i, 7);
        vec_by_scalar_mla_fpred_double_u16_x4(vr2, tmpW_mat2, tmpe, neon_p, i, 7);
    }
    for (int i = 0; i < 4;i++) {
        vr.val[i] = vaddq_u16(vshrq_n_u16(vr.val[i], 7), vandq_u16(vr.val[i], neon_p));
        vr2.val[i] = vaddq_u16(vshrq_n_u16(vr2.val[i], 7), vandq_u16(vr2.val[i], neon_p));
    }
    uint8x16x4_t vres;
    vres.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[0]), vreinterpretq_u8_u16(vr.val[1]));
    vres.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(vr.val[2]), vreinterpretq_u8_u16(vr.val[3]));
    vres.val[2] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[0]), vreinterpretq_u8_u16(vr2.val[1]));
    vres.val[3] = vuzp1q_u8(vreinterpretq_u8_u16(vr2.val[2]), vreinterpretq_u8_u16(vr2.val[3]));
    vst1q_u8_x4(res, vres);
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));
#else
    FZ_DOUBLEPREC e_neon[ROUND_UP(M,EPI16_PER_REG)] = {0};
    for(int i = 0; i < M; i++) {
        e_neon[i] = (FZ_DOUBLEPREC)e[i];
    }
    fz_inf_w_by_fz_matrix_lv5(res, e_neon, (uint16_t *)W_mat);
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));  
#endif
#endif
}
#else
static
void fz_inf_w_by_fz_matrix(FZ_ELEM res[N],
                           const FZ_ELEM e[M],
                           FZ_ELEM W_mat[M][N-M]){

    memset(res,0,(N-M)*sizeof(FZ_ELEM));
    memcpy(res+(N-M),e,M*sizeof(FZ_ELEM));
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N-M; j++){
            res[j] = FZRED_DOUBLE( (FZ_DOUBLEPREC) res[j] +
                                      (FZ_DOUBLEPREC) e[i] *
                                      (FZ_DOUBLEPREC) W_mat[i][j]);
        }
    }
}
#endif

static inline
void fz_vec_sub_m(FZ_ELEM res[M],
                  const FZ_ELEM a[M],
                  const FZ_ELEM b[M]){
    for(int i = 0; i < M; i++){
        res[i]= FZRED_SINGLE( a[i] + FZRED_OPPOSITE(b[i]) );
    }
}

static inline
int is_fz_vec_in_restr_group_m(const FZ_ELEM in[M]){
    int is_in_ok = 1;
    for(int i=0; i<M; i++){
        is_in_ok = is_in_ok && (in[i] < Z);
    }
    return is_in_ok;
}
static inline
void fz_dz_norm_m(FZ_ELEM v[M]){
    for (int i = 0; i < M; i++){
       v[i] = FZ_DOUBLE_ZERO_NORM(v[i]);
    }
}
#endif
