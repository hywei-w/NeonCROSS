#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if __APPLE__
#define __MEDIAN__
#else
#define __MEDIAN__
#endif
#include "cycles.h"

#include "fp_arith.h"
#include "CROSS.h"
#include "csprng_hash.h"
#include "parameters.h"
#include "merkle_tree.h"
#include "seedtree.h"
#include "rng.h"
#include <libgen.h>

#define NTESTS 10000

uint64_t time0, time1;
uint64_t cycles[NTESTS];

void CROSS_sign_verify_speed(int print_tex){
      fprintf(stderr,"Computing number of clock cycles as the average of %d runs\n", NTESTS);
      pk_t pk;
      sk_t sk;
      CROSS_sig_t signature;
      char message[32] = "Signme!!Signme!!Signme!!Signme!";

      WRAP_FUNC("Key generation Cycles",
              cycles, time0, time1,
              CROSS_keygen(&sk,&pk));

      WRAP_FUNC("Signature Cycles",
              cycles, time0, time1,
              CROSS_sign(&sk,message,8,&signature));

      int is_signature_still_ok = 1;
      LOOP_INIT(time0, time1);
      for(size_t i = 0; i < NTESTS; i++){
          BODY_INIT(time0, time1);
          int is_signature_ok = CROSS_verify(&pk,message,8,&signature);
          is_signature_still_ok = is_signature_ok && is_signature_still_ok;
          BODY_TAIL(cycles, time0, time1);
      }
      LOOP_TAIL("Verification Cycles", cycles, time0, time1);
      printf("\n");
      printf("Keygen-Sign-Verify: %s", is_signature_still_ok == 1 ? "functional\n": "not functional\n" );
}

#if defined(HIGH_PERFORMANCE_AARCH64)
void CROSS_func_speed() {
      printf("Computing number of clock cycles as the average of %d runs\n", NTESTS);
      pk_t pk;
      sk_t sk;
      CROSS_sig_t signature;
      char message[32] = "Signme!!Signme!!Signme!!Signme!";
      FP_ELEM v[N-K];
      FP_ELEM in1[N], in2[N];
      FP_ELEM mat_res[N-K], e[N], V_tr[K][N-K];
      FP_DOUBLEPREC testa[NTESTS];
      FP_ELEM testb[NTESTS];

#if defined(HIGH_PERFORMANCE_AARCH64)
#if defined(RSDP)
      FP_ELEM mat_res_neon[N-K] = {0};
      FP_DOUBLEPREC e_neon[N] = {0};
      FP_DOUBLEPREC V_tr_neon[K][ROUND_UP(N-K,EPI16_PER_REG)] = {0};
#else
      FP_ELEM mat_res_neon[N-K] = {0};
      FP_DOUBLEPREC e_neon[N] = {0};
      FP_DOUBLEPREC V_tr_neon[K][ROUND_UP(N-K,EPI32_PER_REG)] = {0};
#endif
#endif
      FZ_ELEM e_inz[N];
      FP_ELEM res[N], chall_1, u_prime[N];
      FZ_ELEM e_inz_neon[N];
      FP_ELEM res_neon[N], chall_1_neon, u_prime_neon[N];
      FP_ELEM vres_neon[N-K];
      FZ_ELEM ve_neon[N];
      FP_ELEM vres[N-K];
      FZ_ELEM ve[N];
      chall_1_neon = rand() % P;
      for(int i=0;i< N; i++) {
            e[i] = i % P;
            e_neon[i] = (FP_DOUBLEPREC)e[i];
            e_inz[i] = i % Z;
            e_inz_neon[i] = e_inz[i];
            u_prime[i] = i % P;
            u_prime_neon[i] = u_prime[i];
            ve[i] = i % Z;
            ve_neon[i] = ve[i];
      }
      for(int i=0;i< K; i++) {
            for(int j=0;j< N-K; j++) {
                  V_tr[i][j] = rand() % P;
                  V_tr_neon[i][j] = (FP_DOUBLEPREC)V_tr[i][j];
            }
      }
      chall_1 = rand() % P;
      chall_1_neon = chall_1;
      FZ_ELEM res_restr[N], res_restr_neon[N];
      fp_vec_by_fp_matrix(mat_res_neon,e_neon,V_tr_neon);
      restr_vec_by_fp_matrix(vres_neon, ve_neon, V_tr_neon);
#if defined(RSDPG)
      FZ_ELEM e_restr[M], e_restr_neon[N];
      FZ_ELEM W_mat[M][N-M];
      FZ_DOUBLEPREC W_mat_neon[M][ROUND_UP(N-M, EPI16_PER_REG)] = {0};
      for(int i=0;i< M; i++) {
            e_restr[i] = i % Z;
            e_restr_neon[i] = e_restr[i];
      }
      for(int i=0;i< M; i++) {
            for(int j=0;j< N-M; j++) {
                  W_mat[i][j] = rand() % Z;
                  W_mat_neon[i][j] = (FZ_DOUBLEPREC)W_mat[i][j];
            }
      }
      fz_inf_w_by_fz_matrix(res_restr_neon, e_restr_neon, W_mat_neon);
#endif
      WRAP_FUNC("fp_vec_by_fp_matrix",
          cycles, time0, time1,
          fp_vec_by_fp_matrix(mat_res_neon,e_neon,V_tr_neon));
      WRAP_FUNC("restr_vec_by_fp_matrix",
          cycles, time0, time1,
          restr_vec_by_fp_matrix(vres_neon, ve_neon, V_tr_neon));
#if defined(RSDPG)
      WRAP_FUNC("fz_inf_w_by_fz_matrix",
          cycles, time0, time1,
          fz_inf_w_by_fz_matrix(res_restr_neon, e_restr_neon, W_mat_neon));
#endif

    unsigned char rounds_seeds[T*SEED_LENGTH_BYTES];
    unsigned char root_seed[SEED_LENGTH_BYTES];
    unsigned char salt[SALT_LENGTH_BYTES];
    unsigned char seed_tree[NUM_NODES_SEED_TREE * SEED_LENGTH_BYTES];
    unsigned char indices_to_publish[T];
    unsigned char stored_seeds[TREE_NODES_TO_STORE*SEED_LENGTH_BYTES]; //unsigned char *stored_seeds;
    uint8_t root[HASH_DIGEST_LENGTH];
    unsigned char tree[NUM_NODES_MERKLE_TREE *HASH_DIGEST_LENGTH];
    unsigned char leaves[T][HASH_DIGEST_LENGTH];
    uint8_t recomputed_leaves[T][HASH_DIGEST_LENGTH];
    uint8_t mtp[HASH_DIGEST_LENGTH*TREE_NODES_TO_STORE];
    uint8_t leaves_to_reveal[T];
#if defined(NO_TREES)
      WRAP_FUNC("seedleaves",
  		cycles, time0, time1,
  		seed_leaves(rounds_seeds, root_seed, salt));
    WRAP_FUNC("tree_root",
          cycles, time0, time1,
          tree_root(root, leaves));
#else
    WRAP_FUNC("gen_seed_tree",
       cycles, time0, time1,
        gen_seed_tree(seed_tree, root_seed, salt));
    WRAP_FUNC("seedleaves",
        cycles, time0, time1,
        seed_leaves(rounds_seeds, seed_tree));
    WRAP_FUNC("tree_root",
  		cycles, time0, time1,
  		tree_root(root, tree, leaves));
#endif
}
#else
static void CROSS_func_speed() {
      printf("Computing number of clock cycles as the average of %d runs\n", NTESTS);
      pk_t pk;
      sk_t sk;
      CROSS_sig_t signature;
      char message[32] = "Signme!!Signme!!Signme!!Signme!";
      FP_ELEM mat_res[N-K] ={0};
      FP_ELEM e[N], V_tr[K][N-K];
      FP_ELEM res[N];
      FP_ELEM vres[N-K];
      FZ_ELEM ve[N];

      for(int i=0;i< N; i++) {
            e[i] = i % P;
            ve[i] = i % Z;
      }
      for(int i=0;i< K; i++) {
            for(int j=0;j< N-K; j++) {
                  V_tr[i][j] = rand() % P;
            }
      }
      fp_vec_by_fp_matrix(mat_res,e,V_tr);
      restr_vec_by_fp_matrix(vres, ve, V_tr);
#if defined(RSDPG)
      FZ_ELEM res_restr[N];
      FZ_ELEM e_restr[M];
      FZ_ELEM W_mat[M][N-M];
      for(int i=0;i< M; i++) {
            e_restr[i] = i % Z;
      }
      for(int i=0;i< M; i++) {
            for(int j=0;j< N-M; j++) {
                  W_mat[i][j] = rand() % Z;
            }
      }
      fz_inf_w_by_fz_matrix(res_restr, e_restr, W_mat);
#endif
      WRAP_FUNC("fp_vec_by_fp_matrix",
          cycles, time0, time1,
          fp_vec_by_fp_matrix(mat_res,e,V_tr));
      WRAP_FUNC("restr_vec_by_fp_matrix",
          cycles, time0, time1,
          restr_vec_by_fp_matrix(vres, ve, V_tr));
#if defined(RSDPG)
      WRAP_FUNC("fz_inf_w_by_fz_matrix",
          cycles, time0, time1,
          fz_inf_w_by_fz_matrix(res_restr, e_restr, W_mat));
#endif

      unsigned char rounds_seeds[T*SEED_LENGTH_BYTES];
      unsigned char root_seed[SEED_LENGTH_BYTES];
      unsigned char salt[SALT_LENGTH_BYTES];
      unsigned char seed_tree[NUM_NODES_SEED_TREE * SEED_LENGTH_BYTES];
      unsigned char indices_to_publish[T];
      unsigned char stored_seeds[TREE_NODES_TO_STORE*SEED_LENGTH_BYTES]; // unsigned char *stored_seeds = 0;
      uint8_t root[HASH_DIGEST_LENGTH];
      unsigned char tree[NUM_NODES_MERKLE_TREE *HASH_DIGEST_LENGTH];
      unsigned char leaves[T][HASH_DIGEST_LENGTH];
      uint8_t recomputed_leaves[T][HASH_DIGEST_LENGTH];
      uint8_t mtp[HASH_DIGEST_LENGTH*TREE_NODES_TO_STORE];
      uint8_t leaves_to_reveal[T];
#if defined(NO_TREES)
      WRAP_FUNC("seedleaves",
              cycles, time0, time1,
              seed_leaves(rounds_seeds, root_seed, salt));
      WRAP_FUNC("tree_root",
            cycles, time0, time1,
            tree_root(root, leaves));
#else
      WRAP_FUNC("gen_seed_tree",
          cycles, time0, time1,
          gen_seed_tree(seed_tree, root_seed, salt));
      WRAP_FUNC("seedleaves",
          cycles, time0, time1,
          seed_leaves(rounds_seeds, seed_tree));
      WRAP_FUNC("tree_root",
                cycles, time0, time1,
                tree_root(root, tree, leaves));
#endif
}
#endif

int main(int argc, char* argv[]){
      csprng_initialize(&platform_csprng_state,
                      (const unsigned char *)"0123456789012345",16,0);
      fprintf(stderr,"CROSS implementation benchmarking tool\n");

      char *executable_name = basename(argv[0]);
      printf("Executable name: %s\n", executable_name);

      // Init performance counter
      init_counter();

      CROSS_func_speed();

      CROSS_sign_verify_speed(0);

      return 0;
}
