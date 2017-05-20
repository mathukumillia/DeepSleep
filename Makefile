CPP = g++
CPPFLAGS = -c -O3
#CPPFLAGS = -c -g
LDFLAGS = -lm

long_timeSVD++: long_timeSVD++.o
	$(CPP) $(LDFLAGS) $^ -o $@

long_timeSVD++.o: long_timeSVD++.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

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
	rm -f *.o SVD++ naive_SVD SVD++ SVD++_probe_output.dta SVD++_qual_output.dta timeSVD++ timeSVD++_probe_output.dta timeSVD++_qual_output.dta long_timeSVD++ long_timeSVD++.o