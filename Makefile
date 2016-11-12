APP = $(PWD)
BIN = $(APP)/bin

INCL = -I./third/gflags/include -I./third/boost

AR = $(APP)/third/gflags/lib/libgflags.a
LIB_PATH = -L /usr/local/lib
LIBS = -lpthread

CC = g++
CXXFLAGS = -std=c++11 -fPIC -Wall -static-libstdc++ -march=native -O3

EXE = $(BIN)/moe
OBJS = src/main.o \
       src/moe.o


.PHONY : all clean

all: $(EXE)

.cc.o:
	$(CC) $(CXXFLAGS) -c $< -o $@ $(INCL)
$(EXE): $(OBJS)
	$(CC) $(CXXFLAGS) $^ -o $@ $(INCL) $(AR) $(LIB_PATH) $(LIBS) 

clean:
	rm -rf $(EXE) $(OBJS)

