#ifndef _BIT_GEN
#define _BIT_GEN

#include <stdint.h>

uint8_t time_resistant_bit_extraction(float* data,
                                      uint16_t data_length,
                                      uint16_t vector_nums,
                                      float absol);
#endif
