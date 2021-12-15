
EXECUTABLE := pca_bit_gen
CC := gcc
CFLAGS := -Iinclude/

CLAPACKFILES = \
	ilaenv.c \
	lsame.c \
	scopy.c \
	slamch.c \
	slansy.c \
	sormtr.c \
	sscal.c \
	sstebz.c \
	sstein.c \
	ssterf.c \
	ssyevx.c \
	ssytrd.c \
	xerbla.c \
	ieeeck.c \
	iparmq.c \
	pow_ri.c \
	s_cmp.c \
	s_copy.c \
	slassq.c \
	r_sqrt.c \
	s_cat.c \
	sormql.c \
	r_log.c \
	slaebz.c \
	slarnv.c \
	slagtf.c \
	sasum.c \
	slagts.c \
	sdot.c \
	saxpy.c \
	isamax.c \
	snrm2.c \
	slanst.c \
	slascl.c \
	slae2.c \
	slapy2.c \
	r_sign.c \
	slacpy.c \

SRCFILES = \
	main.c \
	matrix.c \

CLAPACKDIR = clapack

CLAPACK = $(patsubst %,$(CLAPACKDIR)/%,$(CLAPACKFILES))

SDIR = src

SRC = $(patsubst %,$(SDIR)/%,$(SRCFILES))

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) -g -o $@ $(SRC) $(CLAPACK)

all: $(EXECUTABLE)

clean:
	rm pca_bit_gen

