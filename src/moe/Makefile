LIB=../../third/gflags/lib/libgflags.a

CC = gcc

CXX = g++
CFLAGS = -std=c++11 -fPIC -static-libstdc++ -march=native -DOPENMP -D_GLIBCXX_USE_CXX11_ABI=0 -g -O3 -Wall -msse3 -Wno-unknown-pragmas -funroll-loops -fopenmp \
	 -I../../third/eigen/include \
	 -I../../third/gflags/include \
	 -I../../third/boost \
	 -I../../third


OBJS = main.o moe.o

BIN = moe

.PHONY : all clean
all : $(BIN)

# eigen
#eigen : eigen.o
#	$(CXX) $(CFLAGS) -o eigen eigen.o $(LIB)
# moe
$(BIN) : $(OBJS)
	$(CXX) $(CFLAGS) -o $(BIN) $(OBJS) $(LIB)

#$(OBJS) : %.o : %.cpp
#	$(CXX) -c $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean :
	rm $(BIN) $(OBJS)

#%.o : %.cpp
#	$(CXX) -c $(CFLAGS) $< -O $@
