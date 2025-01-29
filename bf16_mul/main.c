#include <immintrin.h>
#include <memory.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#endif

// -----------------------------------------------

// Explicitly mark the float as 32-bit
typedef float fp32_t;

// To represent BF16, we use uint16_t
typedef uint16_t bf16_t;

// BF16 can be converted as follows (no rounding)
static bf16_t fp32_to_bf16(fp32_t value) {
    const uint16_t v = (*((uint32_t *)&value)) >> 16;
    return v;
}

static fp32_t bf16_to_fp32(bf16_t value) {
    const uint32_t v = ((uint32_t)value) << 16;
    return (*((fp32_t *)&v));
}

// -----------------------------------------------

void init_mat_a(fp32_t a[16][32]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            a[r][c] = r * 0.5 + c * 0.5; // The value you like
        }
    }
}

void init_mat_b(fp32_t b[32][16]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            b[r][c] = r * 0.5 - c * 0.5; // The value you like
        }
    }
}

void print_float16x16(fp32_t m[16][16]) {
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            printf("%f ", m[r][c]);
        }
        printf("\n");
    }
}

// -----------------------------------------------

// Multiply A and B using naive method
void mul_naive(fp32_t c[16][16], fp32_t a[16][32], fp32_t b[32][16]) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < 32; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// -----------------------------------------------
// See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#!=undefined&techs=AMX

typedef struct tile_config_t {
    uint8_t palette_id;         // 0
    uint8_t start_row;          // 1
    uint8_t reserved_2_15[14];  // 2-15: must be zero
    uint16_t colsb[8];          // 16-31
    uint8_t reserved_32_47[16]; // 32-47: must be zero
    uint8_t rows[8];            // 48-55
    uint8_t reserved_56_63[16]; // 56-63: must be zero
} tile_config_t;

// AMX has 8 tiles
#define TILE_0 0
#define TILE_1 1
#define TILE_2 2
#define TILE_3 3
#define TILE_4 4
#define TILE_5 5
#define TILE_6 6
#define TILE_7 7

void init_tile_config() {
    tile_config_t tile = {0};

    tile.palette_id = 1; // This value is always 0 when using AMX
    tile.start_row = 0;

    // tile.colsb (columns in bytes) range: [0, 63]
    // tile.row range: [0, 15]

    // config for fp32_t c[16][16]
    tile.colsb[TILE_0] = 16 * sizeof(fp32_t); // 64
    tile.rows[TILE_0] = 16;

    // config for bf16_t a[16][32]
    tile.colsb[TILE_1] = 32 * sizeof(bf16_t); // 64
    tile.rows[TILE_1] = 16;

    // config for bf16_t b[32][16]
    // The alignment of B in AMX is a bit tricky.
    // The rows of B are divided by 4 byte elements, i.e., BF16 rows are divided by 2 elements
    tile.colsb[TILE_2] = (16 * 2) * sizeof(bf16_t); // 64
    tile.rows[TILE_2] = 32 / 2;                     // 16

    _tile_loadconfig(&tile);
}

// Multiply A and B using AMX
void mul_amx(fp32_t c[16][16], fp32_t a[16][32], fp32_t b[32][16]) {
    bf16_t a16[16][32];
    bf16_t b16[32][16];

#if defined(__AVX512F__)
    // Convert FP32 to BF16 with AVX-512
    for (int r = 0; r < 16; r += 1) {
        // Load the first 16 elements
        __m512 zmm0 = _mm512_loadu_ps(&a[r][0]); // 16 elements of float

        // Load the latter 16 elements
        __m512 zmm1 = _mm512_loadu_ps(&a[r][16]); // 16 elements of float

        // Total 32 elements compressed and converted to bfloat16
        __m512bh bf16_32 = _mm512_cvtne2ps_pbh(zmm1, zmm0);

        // Store the result
        _mm512_storeu_si512(a16[r], bf16_32);
    }

    for (int r = 0; r < 32; r += 2) {
        __m512 zmm0 = _mm512_loadu_ps(b[r]);     // 16 elements of float
        __m512 zmm1 = _mm512_loadu_ps(b[r + 1]); // 16 elements of float
        __m512bh bf16_32 = _mm512_cvtne2ps_pbh(zmm1, zmm0);
        _mm512_storeu_si512(b16[r], bf16_32);
    }
#else
    for (int r = 0; r < 16; r += 1) {
        for (int c = 0; c < 32; c += 1) {
            a16[r][c] = fp32_to_bf16(a[r][c]);
        }
    }

    for (int r = 0; r < 32; r += 1) {
        for (int c = 0; c < 16; c += 1) {
            b16[r][c] = fp32_to_bf16(b[r][c]);
        }
    }
#endif

    bf16_t b16_transformed[16][32];
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            // The rows of B must be divided by 2 elements before load.
            b16_transformed[r / 2][c * 2 + r % 2] = b16[r][c];
        }
    }

    // Load A and B
    _tile_loadd(TILE_1, a16, 32 * sizeof(bf16_t)); // The stride value specifies the row size
    _tile_loadd(TILE_2, b16_transformed, 32 * sizeof(bf16_t));

    // Initialize a tile for C
    _tile_zero(TILE_0);

    // Multiply A and B and accumulate the result to a tile for C
    _tile_dpbf16ps(TILE_0, TILE_1, TILE_2);

    // Store the result back to C
    _tile_stored(TILE_0, c, 16 * sizeof(fp32_t));
}

// -----------------------------------------------

int main() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
        fflush(stdout);
        return 1;
    }
#endif

    fp32_t a[16][32];
    fp32_t b[32][16];

    init_mat_a(a);
    init_mat_b(b);

    fp32_t c_naive[16][16];
    fp32_t c_amx[16][16];

    mul_naive(c_naive, a, b);

    init_tile_config();
    mul_amx(c_amx, a, b);

    printf("----------------------------------------------- Naive result\n");
    print_float16x16(c_naive);
    printf("----------------------------------------------- AMX result\n");
    print_float16x16(c_amx);

    _tile_release(); // Release the AMX state

    return 0;
}
