#include <stdio.h>
int main(void){int f=41176747;int j=34145614;for(int i=0;i<17;i++){j-=(f+7097440);}int d=43203208;for(int i=0;i<31;i++){d+=j;}printf("%d", d);return 0;}