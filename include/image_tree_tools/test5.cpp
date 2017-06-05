#include <iostream>
#include <cstdio>

#include "image_tree_tools.h"

using namespace std;

void print_bin(int k) {
	int res[32] = {0};
	int t = 0;
	while(k) {
		res[t++] = k & 1;
		k >>= 1;
	}
	for(int i=t-1; i>=0; --i) {
		cout << res[i];
		if(i%3==0) cout << " ";
	}
}

int main(int argc, char* argv[]) {
	GeneralOctree<byte> octree;
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " [input.ot]" << endl;
		return 1;
	}
	octree.from_file(argv[1]);

	for (GeneralOctree<byte>::iterator it = octree.begin(); it != octree.end(); ++it) {
		unsigned int key = it->first;
		byte value = it->second;
		print_bin(key);
		int level = octree.compute_level(key);
		cout << ": level = " << level << ", value = "<< (value? 1:0) << endl;
	}
	return 0;
}
