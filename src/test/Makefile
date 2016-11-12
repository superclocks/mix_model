LIB=../../third/gflags/lib/libgflags.a

CC = gcc

CXX = g++
CFLAGS = -std=c++11 -fPIC -static-libstdc++ -march=native -DOPENMP -D_GLIBCXX_USE_CXX11_ABI=0 -g -O3 -Wall -msse3 -Wno-unknown-pragmas -funroll-loops -fopenmp \
	 -I../../third/eigen/include \
	 -I../../third/gflags/include \
	 -I../../third/boost \
	 -I../../third


OBJS = tm.o

BIN = moe

.PHONY : all clean
all : tm eigen

# eigen
eigen : eigen.o
	$(CXX) $(CFLAGS) -o eigen eigen.o $(LIB)
# moe
tm : tm.o
	$(CXX) $(CFLAGS) -o tm tm.o $(LIB)

#$(OBJS) : %.o : %.cpp
#	$(CXX) -c $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean :
	rm $(BIN) $(OBJS)

#%.o : %.cpp
#	$(CXX) -c $(CFLAGS) $< -O $@
