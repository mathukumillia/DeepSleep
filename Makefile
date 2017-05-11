CPP = g++
CPPFLAGS = -c -g -O3
LDFLAGS = -lm

SVD++: optimized_SVD++.o
	$(CPP) $(LDFLAGS) $^ -o $@

SVD++.o: optimized_SVD++.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

optimized_SVD++.cpp: baselinePrediction.h

naive_SVD: optimized_SVD.o
	$(CPP) $(LDFLAGS) $^ -o $@

optimized_SVD.o: optimized_SVD.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

optimized_SVD.cpp: baselinePrediction.h

clean: 
	rm -f *.o SVD++ naive_SVD