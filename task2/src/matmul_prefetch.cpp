// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#ifndef JC
#define JC 64
#endif
#ifndef PF_D
#define PF_D 64
#endif
#ifndef PF_L
#define PF_L _MM_HINT_T0
#endif
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
void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    int kv=(K/8)*8;

    for(int jc=0;jc<N;jc+=JC){
        int jmax;
        if(jc+JC<N)jmax=jc+JC;
        else jmax=N;
        for(int i=0;i<M;++i){
            const float* a = A + static_cast<long>(i) * lda;
            float* c=C + static_cast<long>(i) * ldc;

            int j=jc;
            for(;j+4<=jmax;j+=4){
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
                #if PF_D>0
                    _mm_prefetch(reinterpret_cast<const char*>(b1+p+PF_D),PF_L);
                    _mm_prefetch(reinterpret_cast<const char*>(b2+p+PF_D),PF_L);
                    _mm_prefetch(reinterpret_cast<const char*>(b3+p+PF_D),PF_L); 
                    _mm_prefetch(reinterpret_cast<const char*>(b4+p+PF_D),PF_L);
                #endif
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
            for(;j<jmax;j++){
                const float* b=B + static_cast<long>(j) * ldb;
                breg acc=_mm256_setzero_ps();
                int p=0;
                for(;p<kv;p+=8){
                    #if PF_D>0
                    _mm_prefetch(reinterpret_cast<const char*>(b+p+PF_D),PF_L);
                    #endif
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
}
