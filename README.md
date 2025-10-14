# NeonCROSS 

This source code is the artifact of the paper *"NeonCROSS: Vectorized Implementation of Post-Quantum Signature CROSS on Cortex-A72 and Apple M3"* (IACR TCHES 2026).  

The reference implementation used is based on [CROSS version 2.2](https://github.com/CROSS-signature/CROSS-implementation).

## System Requirements

**Apple M3**  

+ Tested on MacBook Air with M3 SoC  

+ Recommended toolchain:  clang 19.1.3  

**ARM Cortex-A72**  

+ Tested on Raspberry Pi 4 Model B with ARM Cortex-A72 @ 1.5GHz.  
- Recommended host toolchain: aarch64-linux-gnu-gcc 10.5. Direct compilation on Cortex-A72 is possible, but requires manual adjustment of compiler options in CMakeLists.  
- KAT tests involve OpenSSL; compiling on Cortex-A72 with gcc 10.5 is recommended.

## Test and Benchmarking

```bash
mkdir -p build && cd build
cmake ../ -DREFERENCE=3    # build with Neon optimization
# cmake ../ -DREFERENCE=1  # alternatively, build reference implementation
make
./bin/CROSS_test_*
sudo ./bin/CROSS_benchmark_*
```

## KAT Generation and Verification
```bash
cd KAT_Generation
mkdir -p build && cd build
cmake ../ -DREFERENCE=3    # build with Neon optimization
# cmake ../ -DREFERENCE=1  # alternatively, build reference implementation
make
cd ..
chmod +x gen_all_kat.sh  
./gen_all_kat.sh
cd ../KAT
sha512sum -c sha_512_sum_KATs
```

## License

This project (NeonCROSS) is released under Apache-2.0 license. See [LICENSE](LICENSE) for more information.

Some files contain the modified code from [CROSS official repository](https://github.com/CROSS-signature/CROSS-implementation). These codes are released under CC0-1.0 license.

The file feat.S originates from [neon-ntt](https://github.com/neon-ntt/neon-ntt) and is released under MIT license.

The Keccak optimization files contain modified code from [XKCP](https://github.com/XKCP/XKCP), following its original license terms.

The benchmarking files m1cycles.{c,h} are from [Cortex-A implementation](https://github.com/cothan/SABER/blob/master/Cortex-A_Implementation_KEM/m1cycles.c), which are released under Public Domain or Apache-2.0 license.


