#include <iostream>
#include <cstdio>
#include <set>

using namespace std;

void print_set(set<int> cur_set) {
	for(set<int>::iterator it = cur_set.begin(); it != cur_set.end(); ++it) {
		cout << *it;
	}
}

int main() {
	set<int> new_set;
	for(int i = 0; i < 10; ++i) { 
		new_set.insert(i);
	}
	print_set(new_set); cout << endl;
	for(int i = -5; i < 5; ++i) { 
		new_set.insert(i);
	}
	print_set(new_set); cout << endl;
	return 0;
}
