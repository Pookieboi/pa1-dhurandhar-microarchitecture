// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include <cstring>

#ifndef SIMD_W
#define SIMD_W 256
#endif

#if SIMD_W ==256
    typedef __m256 reg;
    #define RLANE 8
    #define RSET _mm256_set1_ps
    #define RLOAD _mm256_loadu_ps
    #define RSTORE _mm256_storeu_ps
    #define RFMA _mm256_fmadd_ps
#else
    typedef __m128 reg;
    #define RLANE 4
    #define RSET _mm_set1_ps
    #define RLOAD _mm_loadu_ps
    #define RSTORE _mm_storeu_ps
    #define RFMA _mm_fmadd_ps
#endif
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    std::memset(out,0,sizeof(float)*H*W);

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const float a=ker[ky*K+kx];
            const reg ra=RSET(a);

            for(int oy=0;oy<H;oy++){
                float* o=out + (oy*W);
                const float*i =in +(oy+ky)*in_stride +kx;
                int ox=0;
                for(;ox+RLANE<=W;ox+=RLANE){
                    reg ri=RLOAD(i+ox);
                    reg ro=RLOAD(o+ox);
                    ro=RFMA(ri,ra,ro);
                    RSTORE(o+ox,ro);

                }
                for(;ox<W;ox++){
                    o[ox]+=i[ox]*a;
                }
            }
        }
    }
}
