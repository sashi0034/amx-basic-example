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

void init_mat_a(int8_t a[8][32]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 8; ++c) {
            a[r][c] = r + c; //
        }
    }
}

void init_mat_b(int8_t b[32][8]) {
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 8; ++c) {
            b[r][c] = r - c;
        }
    }
}

void print_mat8x8(int32_t m[8][8]) {
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            printf("%d ", m[r][c]);
        }
        printf("\n");
    }
}

// -----------------------------------------------

void mul_naive(int32_t c[8][8], const int8_t a[8][32], const int8_t b[32][8]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
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

void init_tile_config() {
    tile_config_t tile = {0};

    tile.palette_id = 1; // This value is always 0 when using AMX
    tile.start_row = 0;

    // config for int8 c[8][8]
    tile.colsb[0] = 8 * sizeof(int32_t);
    tile.rows[0] = 8;

    // config for int8 a[8][32]
    tile.colsb[1] = 32 * sizeof(int8_t);
    tile.rows[1] = 8;

    // config for int8 b[32][8]
    // The alignment of B in AMX is a bit tricky.
    // The rows of B are divided by 4 byte elements.
    tile.colsb[2] = (8 * 4) * sizeof(int8_t); // 32
    tile.rows[2] = 32 / 4;                    // 8
}

// -----------------------------------------------

int main() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
        fflush(stdout);
        return false;
    } else {
        return true;
    }
#endif

    // -----------------------------------------------
}
