#include <iostream>
#include <cstdio>
#include <vector>
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

				key >>= 3;
				OctreeCoord c = instance.compute_coord(key);
				printf("(%d: %d, %d, %d), ", c.l, c.x, c.y, c.z);
				
				key <<= 3;
				c = instance.compute_coord(key);
				printf("(%d: %d, %d, %d), ", c.l, c.x, c.y, c.z);
				
				cout << "\t Neighbors: ";
				vector<unsigned int> neighbors = instance.get_neighbor_keys(key, 2);
				for(vector<unsigned int>::iterator it = neighbors.begin(); it != neighbors.end(); ++it) {
					OctreeCoord c = instance.compute_coord(*it);
					printf("(%d: %d, %d, %d), ", c.l, c.x, c.y, c.z);
				}
				cout << endl;
			}
		}
	}
	return 0;
}
