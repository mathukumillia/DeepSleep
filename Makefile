CPP = g++
CPPFLAGS = -c -g 
LDFLAGS = -lm

run: SVD++
	./SVD++

SVD++: SVD++.o
	$(CPP) $(LDFLAGS) $^ -o $@

SVD++.o: SVD++.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

clean: 
	rm *.o SVD++