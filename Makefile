CPP = g++
CPPFLAGS = -c -g -O3
LDFLAGS = -lm

timeSVD++: timeSVD++.o
	$(CPP) $(LDFLAGS) $^ -o $@

timeSVD++.o: timeSVD++.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

timeSVD++.cpp: baselinePrediction.h

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
	rm -f *.o SVD++ naive_SVD SVD++ SVD++_probe_output.dta SVD++_qual_output.dta timeSVD++ timeSVD++_probe_output.dta timeSVD++_qual_output.dta