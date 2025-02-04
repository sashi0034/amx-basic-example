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
#define INPUT_CH 3

#define OUTPUT_CH 6

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

typedef struct filter_t {
    struct {
        struct {
            int8_t ch[OUTPUT_CH];
        } cols[FILTER_SIZE];
    } rows[FILTER_SIZE];
} filter_t;

void init_input_data(input_data_t *input) {
    for (int r = 0; r < INPUT_ROWS; ++r) {
        for (int c = 0; c < INPUT_COLS; ++c) {
            for (int ich = 0; ich < INPUT_CH; ++ich) {
                input->rows[r].cols[c].ch[ich] = r - c + ich; // The value you like
            }
        }
    }
}

void init_filter_data(filter_t filter[INPUT_CH]) {
    for (int ich = 0; ich < INPUT_CH; ich++) {
        for (int r = 0; r < FILTER_SIZE; ++r) {
            for (int c = 0; c < FILTER_SIZE; ++c) {
                for (int och = 0; och < OUTPUT_CH; ++och) {
                    filter[ich].rows[r].cols[c].ch[och] = ich + r - c - och; // The value you like
                }
            }
        }
    }
}

void print_output_data(const output_data_t *output) {
    for (int r = 0; r < INPUT_ROWS; ++r) {
        for (int c = 0; c < INPUT_COLS; ++c) {
            for (int och = 0; och < OUTPUT_CH; ++och) {
                printf("%d", output->rows[r].cols[c].ch[och]);
                if (och < OUTPUT_CH - 1)
                    printf(", ");
            }
            printf("; ");
        }
        printf("\n");
    }
}

// -----------------------------------------------

void conv_naive(output_data_t *output, const input_data_t *input, const filter_t filter[INPUT_CH]) {
    for (int r = 0; r <= INPUT_ROWS - FILTER_SIZE; ++r) {
        for (int c = 0; c <= INPUT_COLS - FILTER_SIZE; ++c) {
            for (int ich = 0; ich < INPUT_CH; ++ich) {
                output->rows[r].cols[c].ch[ich] = 0;
                for (int fr = 0; fr < FILTER_SIZE; ++fr) {
                    for (int fc = 0; fc < FILTER_SIZE; ++fc) {
                        for (int och = 0; och < OUTPUT_CH; ++och) {
                            output->rows[r].cols[c].ch[och] +=
                                input->rows[r + fr].cols[c + fc].ch[ich] * filter[ich].rows[fr].cols[fc].ch[och];
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

#define TFILTER_ELEMS ((FILTER_SIZE * INPUT_CH) + (4 - (FILTER_SIZE * INPUT_CH) % 4))

#define TFILETER_ROWS (TFILTER_ELEMS / 4)
#define TFILETER_COLS (OUTPUT_CH * 4)

typedef struct tfilter_t {
    struct {
        uint8_t cols[TFILETER_COLS];
    } rows[TFILETER_ROWS];
} tfilter_t;

void transform_filter(tfilter_t tfilter[FILTER_SIZE], const filter_t filter[INPUT_CH]) {
    memset(tfilter, 0, sizeof(tfilter_t) * FILTER_SIZE);

    for (int r = 0; r < FILTER_SIZE; ++r) {
        for (int c = 0; c < FILTER_SIZE; ++c) {
            for (int och = 0; och < OUTPUT_CH; ++och) {
                for (int ich = 0; ich < INPUT_CH; ++ich) {
                    const int c2 = och;
                    const int r2 = c * INPUT_CH + ich;

                    tfilter[r].rows[r2 / 4].cols[c2 * 4 + r2 % 4] = (filter[ich].rows[r].cols[c].ch[och]);
                }
            }
        }
    }
}

// -----------------------------------------------

void conv_amx(output_data_t *output, const input_data_t *input, const filter_t filter[INPUT_CH]) {
    // Load configuraion for convolution
    tile_config_t tile = {0};

    tile.palette_id = 1;
    tile.start_row = 0;

    // config for filter
    tile.colsb[TILE_0] = TFILETER_COLS * sizeof(int8_t);
    tile.rows[TILE_0] = TFILETER_ROWS;

    // config for output data
    tile.colsb[TILE_1] = OUTPUT_CH * sizeof(int32_t);
    tile.rows[TILE_1] = 1;

    // config for input data
    tile.colsb[TILE_2] = TFILTER_ELEMS * sizeof(int8_t);
    tile.rows[TILE_2] = 1;

    _tile_loadconfig(&tile);

    // -----------------------------------------------

    tfilter_t tfilter[FILTER_SIZE];
    transform_filter(tfilter, filter);

    for (int r = 0; r <= INPUT_ROWS - FILTER_SIZE; ++r) {
        for (int c = 0; c <= INPUT_COLS - FILTER_SIZE; ++c) {
            _tile_zero(TILE_1);

            for (int acc = 0; acc < FILTER_SIZE; ++acc) {
                _tile_loadd(TILE_0, &tfilter[acc].rows[0].cols[0], TFILETER_COLS * sizeof(int8_t));
                _tile_loadd(TILE_2, &input->rows[r + acc].cols[c].ch[0], 0);

                _tile_dpbssd(TILE_1, TILE_2, TILE_0);
            }

            _tile_stored(TILE_1, &output->rows[r].cols[c].ch[0], 0);
        }
    }
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

    input_data_t *input;
    input = (input_data_t *)malloc(sizeof(input_data_t));
    init_input_data(input);

    filter_t filter[INPUT_CH];
    init_filter_data(filter);

    output_data_t *output_naive;
    output_naive = (output_data_t *)malloc(sizeof(output_data_t));
    memset(output_naive, 0, sizeof(output_data_t));

    output_data_t *output_amx;
    output_amx = (output_data_t *)malloc(sizeof(output_data_t));
    memset(output_amx, 0, sizeof(output_data_t));

    // -----------------------------------------------

    conv_naive(output_naive, input, filter);
    printf("----------------------------------------------- Naive result\n");
    print_output_data(output_naive);

    conv_amx(output_amx, input, filter);
    printf("----------------------------------------------- AMX result\n");
    print_output_data(output_amx);

    // -----------------------------------------------

    _tile_release(); // Release the AMX state

    return 0;
}
