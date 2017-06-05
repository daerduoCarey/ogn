#include <iostream>
#include <cstdio>
#include "octree.h"

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

int main()
{
	GeneralOctree<int> instance;
	OctreeCoord coord;
	int level = 2;
	int res = pow(2, level);
	for(int i=0; i<res; ++i) {
		for(int j=0; j<res; ++j) {
			for(int k=0; k<res; ++k) {
				coord.x = i; coord.y = j; coord.z = k; coord.l = level;
				unsigned int key = instance.compute_key(coord);
				printf("(%d: %d, %d, %d): key: ", level, i, j, k);
				print_bin(key);

				coord.x = i/2; coord.y = j/2; coord.z = k/2; coord.l = level-1;
				unsigned int key2 = instance.compute_key(coord);
				printf("\t(%d: %d, %d, %d): key: ", level-1, i/2, j/2, k/2);
				print_bin(key2);
				cout << endl;
			}
		}
	}
	return 0;
}
