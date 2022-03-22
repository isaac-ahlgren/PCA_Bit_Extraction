

CC = gcc

CFLAGS = -g -fPIC -shared -Wall -lpthread -lm -lkissfft-float

TARGET = distance_calc.so

CFILES = \
	distance.c \
	threaded_distance.c \
	fft_wrapper.c \
	main.c

all: $(CFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(CFILES)
	mv $(TARGET) ..

clean:
	rm ../$(TARGET)
