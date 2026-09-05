// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

typedef __m256 breg;
typedef __m128 sreg;
inline float rsum(breg v){
    sreg lo=_mm256_castps256_ps128(v);
    sreg hi=_mm256_extractf128_ps(v,1);
    sreg sum=_mm_add_ps(lo,hi);
    sum=_mm_hadd_ps(sum,sum);
    sum=_mm_hadd_ps(sum,sum);
    return _mm_cvtss_f32(sum);
}
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    int kv=(K/8)*8;
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c=C + static_cast<long>(i) * ldc;

        int j=0;
        for(;j+4<=N;j+=4){
            const float* b1 = B + static_cast<long>(j) * ldb;
            const float* b2 = B + static_cast<long>(j+1) * ldb;
            const float* b3 = B + static_cast<long>(j+2) * ldb;
            const float* b4 = B + static_cast<long>(j+3) * ldb;

            breg acc1=_mm256_setzero_ps();
            breg acc2=_mm256_setzero_ps();
            breg acc3=_mm256_setzero_ps();
            breg acc4=_mm256_setzero_ps();

            int p=0;
            for(;p<kv;p+=8){
                breg ra=_mm256_loadu_ps(a+p);

                acc1=_mm256_fmadd_ps(ra,_mm256_loadu_ps(b1+p),acc1);
                acc2=_mm256_fmadd_ps(ra,_mm256_loadu_ps(b2+p),acc2);
                acc3=_mm256_fmadd_ps(ra,_mm256_loadu_ps(b3+p),acc3);
                acc4=_mm256_fmadd_ps(ra,_mm256_loadu_ps(b4+p),acc4);

            }
            float sum1=rsum(acc1);
            float sum2=rsum(acc2);
            float sum3=rsum(acc3);
            float sum4=rsum(acc4);
            for(;p<K;p++){
                float rv=a[p];
                sum1+=rv*b1[p];
                sum2+=rv*b2[p];
                sum3+=rv*b3[p];
                sum4+=rv*b4[p];
            }
            c[j]=sum1;
            c[j+1]=sum2;
            c[j+2]=sum3;
            c[j+3]=sum4;
        }
        for(;j<N;j++){
            const float* b=B + static_cast<long>(j) * ldb;
            breg acc=_mm256_setzero_ps();
            int p=0;
            for(;p<kv;p+=8){
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+p),_mm256_loadu_ps(b+p),acc);
            }
            float sum=rsum(acc);
            for(;p<K;p++){
                sum+=a[p]*b[p];
            }
            c[j]=sum;
        }
    }

}
