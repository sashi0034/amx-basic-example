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

void init_mat_a(int8_t a[16][32]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            a[r][c] = r + c; // The value you like
        }
    }
}

void init_mat_b(int8_t b[32][16]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            b[r][c] = r - c; // The value you like
        }
    }
}

void print_dword16x16(int32_t m[16][16]) {
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%d ", m[r][c]);
        }
        printf("\n");
    }
}

// -----------------------------------------------

// Multiply A and B using naive method
void mul_naive(int32_t c[16][16], int8_t a[16][32], int8_t b[32][16]) {
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

    // config for int8 c[16][16]
    tile.colsb[TILE_0] = 16 * sizeof(int32_t); // 64
    tile.rows[TILE_0] = 16;

    // config for int8 a[16][32]
    tile.colsb[TILE_1] = 32 * sizeof(int8_t); // 32
    tile.rows[TILE_1] = 16;

    // config for int8 b[32][16]
    // The alignment of B in AMX is a bit tricky.
    // The rows of B are divided by 4 byte elements.
    tile.colsb[TILE_2] = (16 * 4) * sizeof(int8_t); // 64
    tile.rows[TILE_2] = 32 / 4;                     // 8

    _tile_loadconfig(&tile);
}

// Multiply A and B using AMX
void mul_amx(int32_t c[16][16], int8_t a[16][32], int8_t b[32][16]) {
    int8_t b_transformed[8][64];
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 16; ++c) {
            // The rows of B must be divided by 4 byte elements before load.
            b_transformed[r / 4][c * 4 + r % 4] = b[r][c];
        }
    }

    // Load A and B
    _tile_loadd(TILE_1, a, 32 * sizeof(int8_t)); // The stride value specifies the row size
    _tile_loadd(TILE_2, b_transformed, (16 * 4) * sizeof(int8_t));

    // Initialize a tile for C
    _tile_zero(TILE_0);

    // Multiply A and B and accumulate the result to a tile for C
    _tile_dpbssd(TILE_0, TILE_1, TILE_2);

    // Store the result back to C
    _tile_stored(TILE_0, c, 16 * sizeof(int32_t));
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

    int8_t a[16][32];
    int8_t b[32][16];

    init_mat_a(a);
    init_mat_b(b);

    int32_t c_naive[16][16];
    int32_t c_amx[16][16];

    mul_naive(c_naive, a, b);

    init_tile_config();
    mul_amx(c_amx, a, b);

    printf("----------------------------------------------- Naive result\n");
    print_dword16x16(c_naive);
    printf("----------------------------------------------- AMX result\n");
    print_dword16x16(c_amx);

    _tile_release(); // Release the AMX state

    return 0;
}
