
EXECUTABLE := pca_bit_gen
CC := gcc
CFLAGS := -Iinclude/
SRCFILES = main.c matrix.c

SDIR = src

SRC = $(patsubst %,$(SDIR)/%,$(SRCFILES))

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) -g -o $@ $(SRC)

all: $(EXECUTABLE)

clean:
	rm pca_bit_gen

