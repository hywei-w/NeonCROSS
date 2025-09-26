cmake_minimum_required(VERSION 3.7)
project(CROSS C ASM)

set(CMAKE_C_STANDARD 11)

add_compile_options(-w)

# Detect Platform
message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor  : ${CMAKE_SYSTEM_PROCESSOR}")

# Platform-specific setup
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Detected Apple macOS")
    set(CC clang)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
          -Wshadow -march=armv8-a -Wpointer-arith -mtune=native -O3 -Wno-unused-result -fomit-frame-pointer -g")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    message(STATUS "Detected Intel x86_64 Linux")
    set(CC aarch64-linux-gnu-gcc-10)
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc-10")
    set(CMAKE_ASM_COMPILER "aarch64-linux-gnu-gcc-10")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -static -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
          -mcpu=cortex-a72 -Wshadow -Wpointer-arith -fomit-frame-pointer -O3 -Wno-unused-result")
else()
    message(WARNING "Unknown platform, please set compiler manually")
endif()

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${SANITIZE}")
message("Compilation flags:" ${CMAKE_C_FLAGS})

# by default, employ neon implementation
if (NOT DEFINED REFERENCE)
    set(REFERENCE 3)
endif()
## employ reference implementation
#set(REFERENCE 1)

find_library(KECCAK_LIB keccak)
if(NOT KECCAK_LIB)
    set(STANDALONE_KECCAK 1)
endif()
set(STANDALONE_KECCAK 1)

# selection of specialized compilation units differing between ref and opt implementations.
set(REFERENCE_CODE_DIR ref)
set(OPTIMIZED_CODE_DIR neon)

if(REFERENCE EQUAL 1)
    message("Compiling portable reference code")
    set(BASE_DIR ${REFERENCE_CODE_DIR})
    set(SPEC_HEADERS
            ${BASE_DIR}/include/sha3.h
            ${BASE_DIR}/include/csprng_hash.h
            ${BASE_DIR}/include/restr_arith.h
            ${BASE_DIR}/include/fp_arith.h
    )
    set(SPEC_SOURCES
            ${BASE_DIR}/lib/CROSS.c
            ${BASE_DIR}/lib/seedtree.c
            ${BASE_DIR}/lib/merkle.c
    )
else()
    message("Compiling optimized NEON code")
    set(BASE_DIR ${OPTIMIZED_CODE_DIR})
    set(SPEC_HEADERS
            ${BASE_DIR}/include/architecture_detect.h
            ${BASE_DIR}/include/fips202x2.h
            ${BASE_DIR}/include/sha3.h
            ${BASE_DIR}/include/csprng_hash.h
            ${BASE_DIR}/include/fp_arith.h
            ${BASE_DIR}/include/restr_arith.h
            ${BASE_DIR}/include/restr_matvec.S
    )
    set(SPEC_SOURCES
            ${BASE_DIR}/lib/CROSS.c
            ${BASE_DIR}/lib/feat.S
            ${BASE_DIR}/lib/fips202x2.c
            ${BASE_DIR}/lib/merkle.c
            ${BASE_DIR}/lib/seedtree.c
    )
endif()

set(COMMON_DIR ${REFERENCE_CODE_DIR})
set(HEADERS
        ${SPEC_HEADERS}
        ${COMMON_DIR}/include/api.h
        ${COMMON_DIR}/include/CROSS.h
        ${COMMON_DIR}/include/pack_unpack.h
        ${COMMON_DIR}/include/fips202.h
        ${COMMON_DIR}/include/keccakf1600.h
        ${COMMON_DIR}/include/parameters.h
        ${COMMON_DIR}/include/seedtree.h
        ${COMMON_DIR}/include/merkle_tree.h
)

if(STANDALONE_KECCAK)
    message("Employing standalone SHA-3")
    set(KECCAK_EXTERNAL_LIB "")
    set(KECCAK_EXTERNAL_ENABLE "")
    list(APPEND FALLBACK_SOURCES ${COMMON_DIR}/lib/keccakf1600.c)
    list(APPEND FALLBACK_SOURCES ${COMMON_DIR}/lib/fips202.c)
else()
    message("Employing libkeccak")
    set(KECCAK_EXTERNAL_LIB keccak)
    set(KECCAK_EXTERNAL_ENABLE "-DSHA_3_LIBKECCAK")
endif()


set(SOURCES
        ${SPEC_SOURCES}
        ${FALLBACK_SOURCES}
        ${COMMON_DIR}/lib/csprng_hash.c
        ${COMMON_DIR}/lib/pack_unpack.c
        ${COMMON_DIR}/lib/keccakf1600.c
        ${COMMON_DIR}/lib/fips202.c
        ${COMMON_DIR}/lib/sign.c
)

foreach(category RANGE 1 5 2)
    set(RSDP_VARIANTS RSDP RSDPG)
    foreach(RSDP_VARIANT ${RSDP_VARIANTS})
        set(PARAM_TARGETS SIG_SIZE BALANCED SPEED)
        foreach(optimiz_target ${PARAM_TARGETS})
            if(optimiz_target STREQUAL SPEED)
                set(OMIT_SEED_TREE "-DNO_TREES=1")
            else()
                set(OMIT_SEED_TREE "")
            endif()
            # settings for benchmarking binary
            set(TARGET_BINARY_NAME CROSS_benchmark_cat_${category}_${RSDP_VARIANT}_${optimiz_target})
            add_executable(${TARGET_BINARY_NAME} ${HEADERS} ${SOURCES} test/include/rng.h
                    test/lib/CROSS_benchmark.c test/lib/cycles.c)
            target_include_directories(${TARGET_BINARY_NAME} PRIVATE
                    ${BASE_DIR}/include
                    ${COMMON_DIR}/include
                    test/include)
            target_link_libraries(${TARGET_BINARY_NAME} m ${SANITIZE} ${KECCAK_EXTERNAL_LIB})
            set_target_properties(${TARGET_BINARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./bin)
            set_property(TARGET ${TARGET_BINARY_NAME} APPEND PROPERTY
                    COMPILE_FLAGS "${OMIT_SEED_TREE} -DCATEGORY_${category}=1 -D${optimiz_target}=1 -D${RSDP_VARIANT}=1 ${KECCAK_EXTERNAL_ENABLE} ")

            # settings for unit tests binary
            set(TARGET_BINARY_NAME CROSS_test_cat_${category}_${RSDP_VARIANT}_${optimiz_target})
            add_executable(${TARGET_BINARY_NAME} ${HEADERS} ${SOURCES} test/include/arith_unit_tests.h test/include/cycles.h test/include/rng.h
                    test/lib/CROSS_test.c)
            target_include_directories(${TARGET_BINARY_NAME} PRIVATE
                    ${BASE_DIR}/include
                    ${COMMON_DIR}/include
                    test/include)
            target_link_libraries(${TARGET_BINARY_NAME} m ${SANITIZE} ${KECCAK_EXTERNAL_LIB})
            set_target_properties(${TARGET_BINARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./bin)
            set_property(TARGET ${TARGET_BINARY_NAME} APPEND PROPERTY
                    COMPILE_FLAGS "${OMIT_SEED_TREE} -DCATEGORY_${category} -D${optimiz_target}=1 -D${RSDP_VARIANT}=1 ${KECCAK_EXTERNAL_ENABLE} ")
        endforeach(optimiz_target)
    endforeach(RSDP_VARIANT)
endforeach(category)

