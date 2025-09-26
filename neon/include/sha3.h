/*
 * NeonCROSS: Vectorized Implementation of the Post-Quantum Signature Algorithm CROSS
 * Copyright (c) 2025 Hanyu Wei et al.
 * Licensed under the Apache License, Version 2.0; see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(SHA_3_LIBKECCAK)
#include <libkeccak.a.headers/KeccakHash.h>

/* LibKeccak SHAKE Wrappers */

#define SHAKE_STATE_STRUCT Keccak_HashInstance
static inline
void xof_shake_init(SHAKE_STATE_STRUCT *state, int val)
{
   if (val == 128)
      /* will result in a zero-length output for Keccak_HashFinal */
      Keccak_HashInitialize_SHAKE128(state);
   else
      /* will result in a zero-length output for Keccak_HashFinal */
      Keccak_HashInitialize_SHAKE256(state);
}

static inline
void xof_shake_update(SHAKE_STATE_STRUCT *state,
                      const unsigned char *input,
                      unsigned int inputByteLen)
{
   Keccak_HashUpdate(state,
                     (const BitSequence *) input,
                     (BitLength) inputByteLen*8 );
}

static inline
void xof_shake_final(SHAKE_STATE_STRUCT *state)
{
   Keccak_HashFinal(state, NULL);
}

static inline
void xof_shake_extract(SHAKE_STATE_STRUCT *state,
                       unsigned char *output,
                       unsigned int outputByteLen)
{
   Keccak_HashSqueeze(state,
                      (BitSequence *) output,
                      (BitLength) outputByteLen*8 );
}

#else
#include "fips202.h"
/* standalone FIPS-202 implementation has 
 * different states for SHAKE depending on security level*/
#if defined(CATEGORY_1)
#define SHAKE_STATE_STRUCT shake128incctx
#else
#define SHAKE_STATE_STRUCT shake256incctx
#endif
// %%%%%%%%%%%%%%%%%% Self-contained SHAKE Wrappers %%%%%%%%%%%%%%%%%%%%%%%%%%%%

static inline
void xof_shake_init(SHAKE_STATE_STRUCT *state, int val)
{
#if defined(CATEGORY_1)
   shake128_inc_init(state);
#else
   shake256_inc_init(state);
#endif
   /* avoid -Werror=unused-parameter */
   (void)val;
}

static inline
void xof_shake_update(SHAKE_STATE_STRUCT *state,
                      const unsigned char *input,
                      unsigned int inputByteLen)
{
#if defined(CATEGORY_1)
   shake128_inc_absorb(state,
                       (const uint8_t *)input,
                       inputByteLen);
#else
   shake256_inc_absorb(state,
                       (const uint8_t *)input,
                       inputByteLen);
#endif
}

static inline
void xof_shake_final(SHAKE_STATE_STRUCT *state)
{
#if defined(CATEGORY_1)
   shake128_inc_finalize(state);
#else
   shake256_inc_finalize(state);
#endif
}

static inline
void xof_shake_extract(SHAKE_STATE_STRUCT *state,
                       unsigned char *output,
                       unsigned int outputByteLen){
#if defined(CATEGORY_1)
   shake128_inc_squeeze(output, outputByteLen, state);
#else
   shake256_inc_squeeze(output, outputByteLen, state);
#endif
}
#endif

#include "architecture_detect.h"
#if defined(HIGH_PERFORMANCE_AARCH64)

// %%%%%%%%%%%%%%%%%% Self-contained SHAKE x2 Wrappers %%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "fips202x2.h"
#define SHAKE_X2_STATE_STRUCT keccakx2_state

static inline void xof_shake_x2_init(SHAKE_X2_STATE_STRUCT *states) {
   keccakx2_inc_init(states);
}
static inline void xof_shake_x2_update(SHAKE_X2_STATE_STRUCT *states,
                      const unsigned char *in0,
                      const unsigned char *in1,
                      uint32_t singleInputByteLen) {
   keccakx2_inc_absorb(states, in0, in1, singleInputByteLen);
}
static inline void xof_shake_x2_final(SHAKE_X2_STATE_STRUCT *states) {
   keccakx2_inc_finalize(states);
}
static inline void xof_shake_x2_extract(SHAKE_X2_STATE_STRUCT *states,
                       unsigned char *out0,
                       unsigned char *out1,
                       uint32_t singleOutputByteLen){
   keccakx2_inc_squeeze(states, out0, out1, singleOutputByteLen);
}
#else
// %%%%%%%%%%%%%%%%%% Self-contained SHAKE x2 Wrappers %%%%%%%%%%%%%%%%%%%%%%%%%%%%

/* SHAKE_x2 just calls SHAKE_x1 twice. If a suitable SHAKE_x2 implementation becomes
 * available, it should be used instead */
typedef struct {
   SHAKE_STATE_STRUCT state1;
   SHAKE_STATE_STRUCT state2;
} shake_x2_ctx;
#define SHAKE_X2_STATE_STRUCT shake_x2_ctx

static inline void xof_shake_x2_init(SHAKE_X2_STATE_STRUCT *states, int val) {
#if defined(CATEGORY_1)
   xof_shake_init(&(states->state1), 128);
   xof_shake_init(&(states->state2), 128);
#else
   xof_shake_init(&(states->state1), 256);
   xof_shake_init(&(states->state2), 256);
#endif
   /* avoid -Werror=unused-parameter */
   (void)val;
}
static inline void xof_shake_x2_update(SHAKE_X2_STATE_STRUCT *states,
                      const unsigned char *in1,
                      const unsigned char *in2,
                      uint32_t singleInputByteLen) {
   xof_shake_update(&(states->state1), (const uint8_t *)in1, singleInputByteLen);
   xof_shake_update(&(states->state2), (const uint8_t *)in2, singleInputByteLen);
}
static inline void xof_shake_x2_final(SHAKE_X2_STATE_STRUCT *states) {
   xof_shake_final(&(states->state1));
   xof_shake_final(&(states->state2));
}
static inline void xof_shake_x2_extract(SHAKE_X2_STATE_STRUCT *states,
                       unsigned char *out1,
                       unsigned char *out2,
                       uint32_t singleOutputByteLen){
   xof_shake_extract(&(states->state1), out1, singleOutputByteLen);
   xof_shake_extract(&(states->state2), out2, singleOutputByteLen);
}
#endif

// %%%%%%%%%%%%%%%%%%%% Parallel SHAKE State Struct %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

typedef struct {
   SHAKE_STATE_STRUCT state1;
   SHAKE_X2_STATE_STRUCT state2;
} par_shake_ctx;
