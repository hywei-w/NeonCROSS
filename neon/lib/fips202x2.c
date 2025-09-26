#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <arm_neon.h>
#include "fips202x2.h"

#ifdef PROFILE_HASHING
#include "cycles.h"
uint64_t keccakx2_cycles;
#endif

#define NROUNDS 24
#define SHA3 1

// Rotate by n bit ((a << offset) ^ (a >> (64-offset)))
#define ROL(a, offset) vsriq_n_u64(vshlq_n_u64(a, offset), a, 64 - offset)

/* Keccak round constants */
static const uint64_t KeccakF_RoundConstants[NROUNDS] = {
    (uint64_t)0x0000000000000001ULL,
    (uint64_t)0x0000000000008082ULL,
    (uint64_t)0x800000000000808aULL,
    (uint64_t)0x8000000080008000ULL,
    (uint64_t)0x000000000000808bULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008009ULL,
    (uint64_t)0x000000000000008aULL,
    (uint64_t)0x0000000000000088ULL,
    (uint64_t)0x0000000080008009ULL,
    (uint64_t)0x000000008000000aULL,
    (uint64_t)0x000000008000808bULL,
    (uint64_t)0x800000000000008bULL,
    (uint64_t)0x8000000000008089ULL,
    (uint64_t)0x8000000000008003ULL,
    (uint64_t)0x8000000000008002ULL,
    (uint64_t)0x8000000000000080ULL,
    (uint64_t)0x000000000000800aULL,
    (uint64_t)0x800000008000000aULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008080ULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008008ULL
};

/*************************************************
* Name:        KeccakF1600_StatePermutex2
*
* Description: The Keccak F1600 Permutation
*
* Arguments:   - uint64_t *state: pointer to input/output Keccak state
**************************************************/
extern void f1600x2(v128 *, const uint64_t *);
static inline
void KeccakF1600_StatePermutex2(v128 state[25]) {
    #if (__APPLE__ && __ARM_FEATURE_CRYPTO) || (__ARM_FEATURE_SHA3) /* although not sure what is being implemented, we find something fast */
    f1600x2(state, KeccakF_RoundConstants);
    #else
    v128 Aba, Abe, Abi, Abo, Abu;
    v128 Aga, Age, Agi, Ago, Agu;
    v128 Aka, Ake, Aki, Ako, Aku;
    v128 Ama, Ame, Ami, Amo, Amu;
    v128 Asa, Ase, Asi, Aso, Asu;
    v128 BCa, BCe, BCi, BCo, BCu; // tmp
    v128 Da, De, Di, Do, Du;      // D
    v128 Eba, Ebe, Ebi, Ebo, Ebu;
    v128 Ega, Ege, Egi, Ego, Egu;
    v128 Eka, Eke, Eki, Eko, Eku;
    v128 Ema, Eme, Emi, Emo, Emu;
    v128 Esa, Ese, Esi, Eso, Esu;

    //copyFromState(A, state)
    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (int round = 0; round < NROUNDS; round += 2) {

        BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        //thetaRhoPiChiIotaPrepareTheta(round, A, E)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Aba ^= Da;
        BCa = Aba;
        Age ^= De;
        BCe = ROL(Age, 44);
        Aki ^= Di;
        BCi = ROL(Aki, 43);
        Amo ^= Do;
        BCo = ROL(Amo, 21);
        Asu ^= Du;
        BCu = ROL(Asu, 14);
        Eba =   BCa ^ ((~BCe)&  BCi );
        Eba ^= vdupq_n_u64(KeccakF_RoundConstants[round]);
        Ebe =   BCe ^ ((~BCi)&  BCo );
        Ebi =   BCi ^ ((~BCo)&  BCu );
        Ebo =   BCo ^ ((~BCu)&  BCa );
        Ebu =   BCu ^ ((~BCa)&  BCe );

        Abo ^= Do;
        BCa = ROL(Abo, 28);
        Agu ^= Du;
        BCe = ROL(Agu, 20);
        Aka ^= Da;
        BCi = ROL(Aka,  3);
        Ame ^= De;
        BCo = ROL(Ame, 45);
        Asi ^= Di;
        BCu = ROL(Asi, 61);
        Ega =   BCa ^ ((~BCe)&  BCi );
        Ege =   BCe ^ ((~BCi)&  BCo );
        Egi =   BCi ^ ((~BCo)&  BCu );
        Ego =   BCo ^ ((~BCu)&  BCa );
        Egu =   BCu ^ ((~BCa)&  BCe );

        Abe ^= De;
        BCa = ROL(Abe,  1);
        Agi ^= Di;
        BCe = ROL(Agi,  6);
        Ako ^= Do;
        BCi = ROL(Ako, 25);
        Amu ^= Du;
        BCo = ROL(Amu,  8);
        Asa ^= Da;
        BCu = ROL(Asa, 18);
        Eka =   BCa ^ ((~BCe)&  BCi );
        Eke =   BCe ^ ((~BCi)&  BCo );
        Eki =   BCi ^ ((~BCo)&  BCu );
        Eko =   BCo ^ ((~BCu)&  BCa );
        Eku =   BCu ^ ((~BCa)&  BCe );

        Abu ^= Du;
        BCa = ROL(Abu, 27);
        Aga ^= Da;
        BCe = ROL(Aga, 36);
        Ake ^= De;
        BCi = ROL(Ake, 10);
        Ami ^= Di;
        BCo = ROL(Ami, 15);
        Aso ^= Do;
        BCu = ROL(Aso, 56);
        Ema =   BCa ^ ((~BCe)&  BCi );
        Eme =   BCe ^ ((~BCi)&  BCo );
        Emi =   BCi ^ ((~BCo)&  BCu );
        Emo =   BCo ^ ((~BCu)&  BCa );
        Emu =   BCu ^ ((~BCa)&  BCe );

        Abi ^= Di;
        BCa = ROL(Abi, 62);
        Ago ^= Do;
        BCe = ROL(Ago, 55);
        Aku ^= Du;
        BCi = ROL(Aku, 39);
        Ama ^= Da;
        BCo = ROL(Ama, 41);
        Ase ^= De;
        BCu = ROL(Ase,  2);
        Esa =   BCa ^ ((~BCe)&  BCi );
        Ese =   BCe ^ ((~BCi)&  BCo );
        Esi =   BCi ^ ((~BCo)&  BCu );
        Eso =   BCo ^ ((~BCu)&  BCa );
        Esu =   BCu ^ ((~BCa)&  BCe );

        //    prepareTheta
        BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Eba ^= Da;
        BCa = Eba;
        Ege ^= De;
        BCe = ROL(Ege, 44);
        Eki ^= Di;
        BCi = ROL(Eki, 43);
        Emo ^= Do;
        BCo = ROL(Emo, 21);
        Esu ^= Du;
        BCu = ROL(Esu, 14);
        Aba =   BCa ^ ((~BCe)&  BCi );
        Aba ^= vdupq_n_u64(KeccakF_RoundConstants[round + 1]);
        Abe =   BCe ^ ((~BCi)&  BCo );
        Abi =   BCi ^ ((~BCo)&  BCu );
        Abo =   BCo ^ ((~BCu)&  BCa );
        Abu =   BCu ^ ((~BCa)&  BCe );

        Ebo ^= Do;
        BCa = ROL(Ebo, 28);
        Egu ^= Du;
        BCe = ROL(Egu, 20);
        Eka ^= Da;
        BCi = ROL(Eka, 3);
        Eme ^= De;
        BCo = ROL(Eme, 45);
        Esi ^= Di;
        BCu = ROL(Esi, 61);
        Aga =   BCa ^ ((~BCe)&  BCi );
        Age =   BCe ^ ((~BCi)&  BCo );
        Agi =   BCi ^ ((~BCo)&  BCu );
        Ago =   BCo ^ ((~BCu)&  BCa );
        Agu =   BCu ^ ((~BCa)&  BCe );

        Ebe ^= De;
        BCa = ROL(Ebe, 1);
        Egi ^= Di;
        BCe = ROL(Egi, 6);
        Eko ^= Do;
        BCi = ROL(Eko, 25);
        Emu ^= Du;
        BCo = ROL(Emu, 8);
        Esa ^= Da;
        BCu = ROL(Esa, 18);
        Aka =   BCa ^ ((~BCe)&  BCi );
        Ake =   BCe ^ ((~BCi)&  BCo );
        Aki =   BCi ^ ((~BCo)&  BCu );
        Ako =   BCo ^ ((~BCu)&  BCa );
        Aku =   BCu ^ ((~BCa)&  BCe );

        Ebu ^= Du;
        BCa = ROL(Ebu, 27);
        Ega ^= Da;
        BCe = ROL(Ega, 36);
        Eke ^= De;
        BCi = ROL(Eke, 10);
        Emi ^= Di;
        BCo = ROL(Emi, 15);
        Eso ^= Do;
        BCu = ROL(Eso, 56);
        Ama =   BCa ^ ((~BCe)&  BCi );
        Ame =   BCe ^ ((~BCi)&  BCo );
        Ami =   BCi ^ ((~BCo)&  BCu );
        Amo =   BCo ^ ((~BCu)&  BCa );
        Amu =   BCu ^ ((~BCa)&  BCe );

        Ebi ^= Di;
        BCa = ROL(Ebi, 62);
        Ego ^= Do;
        BCe = ROL(Ego, 55);
        Eku ^= Du;
        BCi = ROL(Eku, 39);
        Ema ^= Da;
        BCo = ROL(Ema, 41);
        Ese ^= De;
        BCu = ROL(Ese, 2);
        Asa =   BCa ^ ((~BCe)&  BCi );
        Ase =   BCe ^ ((~BCi)&  BCo );
        Asi =   BCi ^ ((~BCo)&  BCu );
        Aso =   BCo ^ ((~BCu)&  BCa );
        Asu =   BCu ^ ((~BCa)&  BCe );

    }

    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
    #endif
}

/* ---------------------------------------------------------------- */
#define SnP_laneLengthInBytes 8
// #define SnP_log_laneLengthInBytes 3
// #define SnP_and_laneLengthInBytes 0x7
#define laneIndex(instanceIndex, lanePosition) ((lanePosition<<1) + instanceIndex)

#define LOAD2_64(low, high) vcombine_u64(vcreate_u64(low), vcreate_u64(high))
#define LOAD6464(a, b) vcombine_u64(vcreate_u64(b), vcreate_u64(a)) // high: a, low: b
#define XOReq128(a, b) ((a) = veorq_u64((a), (b)))

#define LOAD128(a)          vld1q_u64(a)
#define STORE128(a, b)      vst1q_u64((uint64_t *)(a), b)
#define STORE64L(a, b)      vst1_u64(a, vget_low_u64(b))
#define STORE64H(a, b)      vst1_u64(a, vget_high_u64(b))

#define UNPACKL(a, b)  vzip1q_u64((a), (b))  // { a[0], b[0] }
#define UNPACKH(a, b)  vzip2q_u64((a), (b))  // { a[1], b[1] }

/* ---------------------------------------------------------------- */
void KeccakP1600times2_AddBytes(keccakx2_state *state, unsigned int instanceIndex, const unsigned char *data, unsigned int offset, unsigned int length) {
    unsigned int sizeLeft = length;
    unsigned int lanePosition = offset/SnP_laneLengthInBytes;
    unsigned int offsetInLane = offset%SnP_laneLengthInBytes;
    const unsigned char *curData = data;
    uint64_t *statesAsLanes = (uint64_t *)state->s;

    if ((sizeLeft > 0) && (offsetInLane != 0)) {
        unsigned int bytesInLane = SnP_laneLengthInBytes - offsetInLane;
        uint64_t lane = 0;
        if (bytesInLane > sizeLeft)
            bytesInLane = sizeLeft;
        memcpy((unsigned char*)&lane + offsetInLane, curData, bytesInLane);
        statesAsLanes[laneIndex(instanceIndex, lanePosition)] ^= lane;
        sizeLeft -= bytesInLane;
        lanePosition++;
        curData += bytesInLane;
    }

    while(sizeLeft >= SnP_laneLengthInBytes) {
        uint64_t lane = *((const uint64_t*)curData);
        statesAsLanes[laneIndex(instanceIndex, lanePosition)] ^= lane;
        sizeLeft -= SnP_laneLengthInBytes;
        lanePosition++;
        curData += SnP_laneLengthInBytes;
    }

    if (sizeLeft > 0) {
        uint64_t lane = 0;
        memcpy(&lane, curData, sizeLeft);
        statesAsLanes[laneIndex(instanceIndex, lanePosition)] ^= lane;
    }
}

/* ---------------------------------------------------------------- */
void KeccakP1600times2_AddLanesAll(keccakx2_state *state, const unsigned char *data, unsigned int laneCount, unsigned int laneOffset)
{
    v128 *stateAsLanes = state->s;
    unsigned int i;
    const uint64_t *curData0 = (const uint64_t *)data;
    const uint64_t *curData1 = (const uint64_t *)(data+laneOffset*SnP_laneLengthInBytes);
    #define XOR_In( argIndex )  XOReq128( stateAsLanes[argIndex], LOAD6464(curData1[argIndex], curData0[argIndex]))
    if ( laneCount >= 17 )  {
        XOR_In( 0 );
        XOR_In( 1 );
        XOR_In( 2 );
        XOR_In( 3 );
        XOR_In( 4 );
        XOR_In( 5 );
        XOR_In( 6 );
        XOR_In( 7 );
        XOR_In( 8 );
        XOR_In( 9 );
        XOR_In( 10 );
        XOR_In( 11 );
        XOR_In( 12 );
        XOR_In( 13 );
        XOR_In( 14 );
        XOR_In( 15 );
        XOR_In( 16 );
        if ( laneCount >= 21 )  {
            XOR_In( 17 );
            XOR_In( 18 );
            XOR_In( 19 );
            XOR_In( 20 );
            for(i=21; i<laneCount; i++)
                XOR_In( i );
        }
        else {
            for(i=17; i<laneCount; i++)
                XOR_In( i );
        }
    }
    else {
        for(i=0; i<laneCount; i++)
            XOR_In( i );
    }
    #undef  XOR_In
}

/* ---------------------------------------------------------------- */
void KeccakP1600times2_ExtractLanesAll(const keccakx2_state *states, unsigned char *data, unsigned int laneCount, unsigned int laneOffset) {
    const uint64_t *stateAsLanes = (const uint64_t *)states->s;
    v128 lanes;
    unsigned int i;
    uint64_t *curData0 = (uint64_t *)data;
    uint64_t *curData1 = (uint64_t *)(data+laneOffset*SnP_laneLengthInBytes);
    #define Extr( argIndex )    lanes = LOAD128(&stateAsLanes[argIndex<<1]),          \
                                STORE64L(&curData0[argIndex], lanes ),              \
                                STORE64H(&curData1[argIndex], lanes )
    #define Extr2( argIndex )   lanes0 = LOAD128(&stateAsLanes[argIndex<<1]),         \
                                lanes1 = LOAD128(&stateAsLanes[(argIndex+1)<<1]),     \
                                lanes =  UNPACKL( lanes0, lanes1 ),                 \
                                lanes0 = UNPACKH( lanes0, lanes1 ),                 \
                                STORE128(&curData0[argIndex], lanes ),    \
                                STORE128(&curData1[argIndex], lanes0)
    if ( laneCount >= 16 )  {
        v128 lanes0, lanes1;
        Extr2( 0 );
        Extr2( 2 );
        Extr2( 4 );
        Extr2( 6 );
        Extr2( 8 );
        Extr2( 10 );
        Extr2( 12 );
        Extr2( 14 );
        if ( laneCount >= 20 )  {
            Extr2( 16 );
            Extr2( 18 );
            for(i=20; i<laneCount; i++)
                Extr( i );
        }
        else {
            for(i=16; i<laneCount; i++)
                Extr( i );
        }
    }
    #undef  Extr2
    else {
        for(i=0; i<laneCount; i++)
            Extr( i );
    }
    #undef  Extr
}

void KeccakP1600times2_ExtractBytes(keccakx2_state *states, unsigned int instanceIndex, unsigned char *data, unsigned int offset, unsigned int length)
{
    unsigned int sizeLeft = length;
    unsigned int lanePosition = offset/SnP_laneLengthInBytes;
    unsigned int offsetInLane = offset%SnP_laneLengthInBytes;
    unsigned char *curData = data;
    const uint64_t *statesAsLanes = (const uint64_t *)states->s;

    if ((sizeLeft > 0) && (offsetInLane != 0)) {
        unsigned int bytesInLane = SnP_laneLengthInBytes - offsetInLane;
        if (bytesInLane > sizeLeft)
            bytesInLane = sizeLeft;
        memcpy( curData, ((unsigned char *)&statesAsLanes[laneIndex(instanceIndex, lanePosition)]) + offsetInLane, bytesInLane);
        sizeLeft -= bytesInLane;
        lanePosition++;
        curData += bytesInLane;
    }

    while(sizeLeft >= SnP_laneLengthInBytes) {
        *(uint64_t*)curData = statesAsLanes[laneIndex(instanceIndex, lanePosition)];
        sizeLeft -= SnP_laneLengthInBytes;
        lanePosition++;
        curData += SnP_laneLengthInBytes;
    }

    if (sizeLeft > 0) {
        memcpy( curData, &statesAsLanes[laneIndex(instanceIndex, lanePosition)], sizeLeft);
    }
}

/* ---------------------------keccakx2_inc------------------------------ */
void keccakx2_inc_init(keccakx2_state *state)
{
#ifdef PROFILE_HASHING
    uint64_t t0 = get_cycle();
#endif
    /* zero the state */
    memset(state, 0, sizeof(keccakx2_state));
    state->offset = 0;
#ifdef PROFILE_HASHING
    uint64_t t1 = get_cycle();
    keccakx2_cycles += (t1 - t0);
#endif
}

void keccakx2_inc_absorb(keccakx2_state *state,
                         const uint8_t *in0,
                         const uint8_t *in1,
                         size_t in_len)
{
#ifdef PROFILE_HASHING
    uint64_t t0 = get_cycle();
#endif
    unsigned int total_len = 2 * in_len;
    unsigned char *ins = (unsigned char *)malloc(total_len * sizeof(unsigned char));
    assert(ins != NULL);
    uint8_t* original_ins = ins;
    memcpy(ins, in0, in_len);
    memcpy(ins + in_len, in1, in_len);

    if (state->offset == 0 && in_len % WORD_BYTES == 0) {
        int lanes = in_len * 8 / WORD;
        int lane_offset = lanes;
        while (lanes > 0) {
            if (lanes >= MAX_LANES) {
                KeccakP1600times2_AddLanesAll(state, ins, MAX_LANES, lane_offset);
                KeccakF1600_StatePermutex2(state->s);
                lanes -= MAX_LANES;
                ins += MAX_LANES * WORD_BYTES;
                state->offset = 0;
            } else {
                KeccakP1600times2_AddLanesAll(state, ins, lanes, lane_offset);
                state->offset = lanes * WORD_BYTES;
                lanes = 0;
            }
        }
    } else {
        while (in_len + state->offset >= RATE) {
            for (int instance = 0; instance < 2; instance++) {
                KeccakP1600times2_AddBytes(state,
                                           instance,
                                           ins + instance * (total_len/2),
                                           state->offset,
                                           RATE - state->offset);
            }
            in_len -= (RATE - state->offset);
            ins += (RATE - state->offset);
            KeccakF1600_StatePermutex2(state->s);
            state->offset = 0;
        }

        for (int instance = 0; instance < 2; instance++) {
            KeccakP1600times2_AddBytes(state,
                                       instance,
                                       ins + instance * (total_len/2),
                                       state->offset,
                                       in_len);
        }
        state->offset += in_len;
    }

    free(original_ins);
#ifdef PROFILE_HASHING
    uint64_t t1 = get_cycle();
    keccakx2_cycles += (t1 - t0);
#endif
}

void keccakx2_inc_finalize(keccakx2_state *state)
{
#ifdef PROFILE_HASHING
    uint64_t t0 = get_cycle();
#endif
    uint8_t ds = DS;
    if(state->offset == RATE - 1) {
        ds |= 128;
        for(int instance=0; instance<2; instance++) {
            KeccakP1600times2_AddBytes(state, instance, &ds, state->offset, 1);
        }
    } else {
        for(int instance=0; instance<2; instance++) {
            KeccakP1600times2_AddBytes(state, instance, &ds, state->offset, 1);
        }
        ds = 128;
        for(int instance=0; instance<2; instance++) {
            KeccakP1600times2_AddBytes(state, instance, &ds, RATE - 1, 1);
        }
    }
    state->offset = 0;
#ifdef PROFILE_HASHING
    uint64_t t1 = get_cycle();
    keccakx2_cycles += (t1 - t0);
#endif
}

void keccakx2_inc_squeeze(keccakx2_state *state, unsigned char *out0, unsigned char *out1, unsigned int out_len)
{
#ifdef PROFILE_HASHING
    uint64_t t0 = get_cycle();
#endif
    unsigned int total_len = 2 * out_len;
    unsigned char *outs = (unsigned char *)malloc(total_len * sizeof(unsigned char));
    uint8_t* original_outs = outs;
    int original_out_len = out_len;

    if (state->offset == 0 && out_len % WORD_BYTES == 0) {
        int lanes = out_len * 8 / WORD;
        int lane_offset = lanes;
        while (lanes > 0) {
            KeccakF1600_StatePermutex2(state->s);
            if (lanes >= MAX_LANES) {
                KeccakP1600times2_ExtractLanesAll(state, outs, MAX_LANES, lane_offset);
                lanes -= MAX_LANES;
                outs += MAX_LANES * WORD_BYTES;
                state->offset = 0;
            } else {
                KeccakP1600times2_ExtractLanesAll(state, outs, lanes, lane_offset);
                state->offset = RATE - (lanes * WORD_BYTES);
                lanes = 0;
            }
        }
    } else {
        unsigned int len = (out_len < state->offset) ? out_len : state->offset;

        for (int instance = 0; instance < 2; instance++) {
            KeccakP1600times2_ExtractBytes(state, instance,
                outs + instance * (total_len / 2), RATE - state->offset, len);
        }
        outs += len;
        out_len -= len;
        state->offset -= len;

        while (out_len > 0) {
            KeccakF1600_StatePermutex2(state->s);
            len = (out_len < RATE) ? out_len : RATE;
            for (int instance = 0; instance < 2; instance++) {
                KeccakP1600times2_ExtractBytes(state, instance,
                outs + instance * (total_len / 2), 0, len);
            }
            outs += len;
            out_len -= len;
            state->offset = RATE - len;
        }
    }
    memcpy(out0, original_outs, (total_len/2));
    memcpy(out1, original_outs+original_out_len, (total_len/2));
    free(original_outs);
#ifdef PROFILE_HASHING
    uint64_t t1 = get_cycle();
    keccakx2_cycles += (t1 - t0);
#endif
}

