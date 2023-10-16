# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -cxx=icpx -fsycl -fsycl-targets=spir64

LD_FLAGS = -fopenmp -fsycl -lze_loader
CMPIFLAGS =
CMPILIBFLAGS = 

TARGETS = ExaComm
OBJECTS = main.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CXX} ${CXXFLAGS} ${CMPIFLAGS} $< -c -o $@

ExaComm: $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(CMPILIBFLAGS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
