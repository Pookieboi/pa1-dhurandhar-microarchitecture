// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    
    for (int ky = 0; ky < K; ++ky) {
        const float* krow=ker+ky*K;

        for (int oy = 0; oy < H; ++oy) {
            float* o=out+oy*W;
            const float* i=in+(oy+ky)*in_stride;

            for (int ox = 0; ox < W; ox+=8) {
                float a[8];
                if(ky==0){
                    a[0]=a[1]=a[2]=a[3]=a[4]=a[5]=a[6]=a[7]=0.0f;    
                }
                else{
                    a[0]=o[ox+0];
                    a[1]=o[ox+1];
                    a[2]=o[ox+2];
                    a[3]=o[ox+3];
                    a[4]=o[ox+4];
                    a[5]=o[ox+5];
                    a[6]=o[ox+6];
                    a[7]=o[ox+7];
                }
            for (int kx = 0; kx < K; ++kx) {
                const float w=krow[kx];
                const float* q=i+ox+kx;
                a[0]+=q[0]*w;
                a[1]+=q[1]*w;
                a[2]+=q[2]*w;
                a[3]+=q[3]*w;
                a[4]+=q[4]*w;
                a[5]+=q[5]*w;
                a[6]+=q[6]*w;
                a[7]+=q[7]*w;
            }

            o[ox+0]=a[0];
            o[ox+1]=a[1];
            o[ox+2]=a[2];
            o[ox+3]=a[3];
            o[ox+4]=a[4];
            o[ox+5]=a[5];
            o[ox+6]=a[6];
            o[ox+7]=a[7];
            

        }
    }
}
}
