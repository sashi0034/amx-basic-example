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

#define INPUT_ROWS 160
#define INPUT_COLS 160
#define INPUT_CH 4

#define OUTPUT_CH 8

#define FILTER_SIZE 3

typedef struct input_data_t {
    struct {
        struct {
            int8_t ch[INPUT_CH];
        } cols[INPUT_COLS];
    } rows[INPUT_ROWS];
} input_data_t;

typedef struct output_data_t {
    struct {
        struct {
            int32_t ch[OUTPUT_CH];
        } cols[INPUT_COLS];
    } rows[INPUT_ROWS];
} output_data_t;

typedef struct filter_data_t {
    struct {
        struct {
            int8_t ch[OUTPUT_CH];
        } cols[FILTER_SIZE];
    } rows[FILTER_SIZE];
} filter_data_t;

void init_input_data(input_data_t *input) {
    for (int r = 0; r < INPUT_ROWS; ++r) {
        for (int c = 0; c < INPUT_COLS; ++c) {
            for (int ic = 0; ic < INPUT_CH; ++ic) {
                input->rows[r].cols[c].ch[ic] = r - c + ic; // The value you like
            }
        }
    }
}

void init_filter_data(filter_data_t filter[INPUT_CH]) {
    for (int ic = 0; ic < INPUT_CH; ic++) {
        for (int r = 0; r < FILTER_SIZE; ++r) {
            for (int c = 0; c < FILTER_SIZE; ++c) {
                for (int oc = 0; oc < OUTPUT_CH; ++oc) {
                    filter[ic].rows[r].cols[c].ch[oc] = ic + r - c - oc; // The value you like
                }
            }
        }
    }
}

// -----------------------------------------------

void conv_naive(output_data_t *output, const input_data_t *input, const filter_data_t filter[INPUT_CH]) {
    for (int r = 0; r < INPUT_ROWS; ++r) {
        for (int c = 0; c < INPUT_COLS; ++c) {
            for (int ic = 0; ic < INPUT_CH; ++ic) {
                output->rows[r].cols[c].ch[ic] = 0;
                for (int fr = 0; fr < FILTER_SIZE; ++fr) {
                    for (int fc = 0; fc < FILTER_SIZE; ++fc) {
                        for (int oc = 0; oc < OUTPUT_CH; ++oc) {
                            output->rows[r].cols[c].ch[oc] +=
                                input->rows[r + fr].cols[c + fc].ch[ic] * filter[ic].rows[fr].cols[fc].ch[oc];
                        }
                    }
                }
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

void conv_amx(output_data_t *output, const input_data_t *input, const filter_data_t filter[INPUT_CH]) {
    // TODO
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
