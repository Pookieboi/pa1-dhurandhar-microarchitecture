// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include<cstring>
#ifndef TH
#define TH 4
#endif
#ifndef TW
#define TW 1024
#endif
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
              const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    std::memset(out,0,sizeof(float)*H*W);
    for(int ty=0;ty<H;ty+=TH){
        int ymax;
        if(ty+TH<H) ymax=ty+TH;
        else ymax=H;
        for(int tx=0;tx<W;tx+=TW){
            int xmax;
            if(tx+TW<W) xmax=tx+TW;
            else xmax=W;
            for(int ky=0;ky<K;ky++){
                for(int kx=0;kx<K;kx++){
                    float a=ker[ky*K+kx];
                    for(int oy=ty;oy<ymax;oy++){
                        float *o=out+oy*W;
                        const float *i=in+(oy+ky)*in_stride +kx;
                        for(int ox=tx;ox<xmax;ox++){
                            o[ox]+=i[ox]*a;
                        }
                    }
                }
            }
        }
    }
}
