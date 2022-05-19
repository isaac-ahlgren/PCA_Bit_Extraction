

CC = gcc

CFLAGS = -g -fPIC -shared -D__X86_SSE -O0 -march=native -Wall -lpthread -lm
TEST_CFLAGS = -g -Wall -lpthread -lm

TARGET = distance_calc.so
TEST_TARGET = test.o

KISS_FFT = ./libkissfft-float.a

CFILES = \
	threaded_pca_calc.c \
	fft_wrapper.c \
	pca_wrapper.c \
	eig_vec_decomp.c \
	main.c

all: $(CFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(CFILES) $(KISS_FFT)
	mv $(TARGET) ..

test: $(CFILES)
	$(CC) $(TEST_CFLAGS) -o $(TEST_TARGET) $(CFILES) $(KISS_FFT)

clean:
	rm ../$(TARGET)
