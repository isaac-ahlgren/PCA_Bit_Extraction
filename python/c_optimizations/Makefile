

CC = gcc

CFLAGS = -g -fPIC -shared -Wall -lpthread -lm -lkissfft-float

TARGET = distance_calc.so

LIBS = \
	liblapack.a \
	libblas.a \
	libcblaswr.a \
	libf2c.a

CFILES = \
	distance.c \
	threaded_distance.c \
	threaded_pca_calc.c \
	fft_wrapper.c \
	pca_wrapper.c \
	main.c

all: $(CFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(CFILES) $(LIBS)
	mv $(TARGET) ..

clean:
	rm ../$(TARGET)
