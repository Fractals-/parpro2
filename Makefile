CC = mpicc
CXX = mpic++
CFLAGS = -Wall -Wextra -O3 -g -mcmodel=medium

all:	mst

mst:	mst.o mmio.o matrix.o component.o
	$(CXX) $(CFLAGS) -o $@ $^ #-lrt


mmio.o:	mmio.c
	$(CC) $(CFLAGS) -c mmio.c

%.o:	%.cc
	$(CXX) $(CFLAGS) -c $<

clean:
	rm -f *o
	rm -f mst
