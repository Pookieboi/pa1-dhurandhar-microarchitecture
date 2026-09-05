// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"
#ifndef TH
#define TH 16
#endif

#ifndef TW
#define TW 512
#endif

#ifndef NACC 
#define NACC 2
#endif 
typedef __m256 reg;
void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int cols=8*NACC;
    for(int ty=0;ty<H;ty+=TH){
        int ymax;
        if(ty+TH<H) ymax=ty+TH;
        else ymax=H;

        for(int tx=0;tx<W;tx+=TW){
            int xmax;
            if(tx+TW<W) xmax=tx+TW;
            else xmax=W;

            for (int oy = ty; oy < ymax; ++oy) {
                float* o=out+oy*W;
                const float* i=in+(oy)*in_stride;
                int ox=tx;

                for(;ox+cols<=xmax;ox+=cols){
                    reg acc[NACC];
                    for(int a=0;a<NACC;++a) acc[a]=_mm256_setzero_ps();
                    for(int ky=0;ky<K;++ky){
                        const float* wow=i+ky*in_stride +ox;
                        const float* krow=ker+ky*K;
                        
                        for(int kx=0;kx<K;kx++){
                            reg w=_mm256_set1_ps(krow[kx]);
                            for(int j=0;j<NACC;j++){
                                acc[j]=_mm256_fmadd_ps(_mm256_loadu_ps(wow+kx+8*j),w,acc[j]);
                            }
                        }
                    }
                    for(int j=0;j<NACC;++j){
                        _mm256_storeu_ps(o+ox+8*j,acc[j]);
                    }
                }
                for(;ox+8<=xmax;ox+=8){
                    reg acc=_mm256_setzero_ps();
                    for(int ky=0;ky<K;++ky){
                        const float* wow=i+ky*in_stride +ox;
                        const float* krow=ker+ky*K;
                        
                        for(int kx=0;kx<K;kx++){
                            acc=_mm256_fmadd_ps(_mm256_loadu_ps(wow+kx),_mm256_set1_ps(krow[kx]),acc);
                        }
                    }
                    _mm256_storeu_ps(o+ox,acc);
                }

                for(;ox<xmax;++ox){
                    float a=0.0f;
                    for(int ky=0;ky<K;++ky){
                        for(int kx=0;kx<K;++kx){
                            a+=i[ky*in_stride+ox+ky]*ker[ky*K+kx];
                        }
                    }
                    o[ox]=a;
                }
            }
        }
    }
}
