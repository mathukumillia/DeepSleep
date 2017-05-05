CPP = g++
CPPFLAGS = -c -g 
LDFLAGS = -lm

SVD++: SVD++.o
	$(CPP) $(LDFLAGS) $^ -o $@

SVD++.o: SVD++.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

naive_SVD: optimized_SVD.o
	$(CPP) $(LDFLAGS) $^ -o $@

optimized_SVD.o: optimized_SVD.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

clean: 
	rm -f *.o SVD++ naive_SVD