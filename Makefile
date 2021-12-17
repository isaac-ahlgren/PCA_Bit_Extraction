
EXECUTABLE := pca_bit_gen
CC := gcc
CFLAGS := -lm -Iinclude/

LIBFILES = \
	lapack_LINUX.a \
	blas_LINUX.a \
	libfblaswr.a \
	libf2c.a \


SRCFILES = \
	main.c \
	matrix.c \

LIBDIR = libs

LIBS = $(patsubst %,$(LIBDIR)/%,$(LIBFILES))

SDIR = src

SRC = $(patsubst %,$(SDIR)/%,$(SRCFILES))

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) -g -o $@ $(SRC) $(LIBS)

all: $(EXECUTABLE)

clean:
	rm pca_bit_gen

