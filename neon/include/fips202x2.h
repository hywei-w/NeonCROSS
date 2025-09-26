#ifndef FIPS202X2_H
#define FIPS202X2_H

#include <stddef.h>
#include <arm_neon.h>
#include "fips202.h"

typedef uint64x2_t v128;
/************************************************
 *  Function Prototypes
 ***********************************************/
typedef struct {
    v128 s[25];
    uint64_t offset;
    /* - during absrbtion: "offset" is the number of absorbed bytes that have already been xored into the state but have not been permuted yet
     * - during squeezing: "offset" is the number of not-yet-squeezed bytes */
} keccakx2_state;

/* CROSS category 1 uses SHAKE128 */
#if defined(CATEGORY_1)
#define RATE SHAKE128_RATE
#else
#define RATE SHAKE256_RATE
#endif

/* SHAKE Domain Separator */
#define DS (0x1F)
#define WORD (64)
#define WORD_BYTES (WORD / 8)
#define MAX_LANES (RATE / (WORD / 8))

void keccakx2_inc_init(keccakx2_state *state);
void keccakx2_inc_absorb(keccakx2_state *state,
                         const uint8_t *in0,
                         const uint8_t *in1,
                         size_t inlen);
void keccakx2_inc_finalize(keccakx2_state *state);
void keccakx2_inc_squeeze(keccakx2_state *state,
                        unsigned char *out0,
                        unsigned char *out1,
                        unsigned int outlen);
#endif
