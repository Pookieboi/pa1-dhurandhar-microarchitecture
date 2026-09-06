#include<immintrin.h>
#include "matmul.h"

#ifndef JC
#define JC 32
#endif

#ifndef PF_D
#define PF_D 0
#endif

#ifndef PF_L
#define PF_L _MM_HINT_T0
#endif

static inline float hsum256 (__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_scalar (const float* a, const float* b, int K)
{
    const int Kv = K & ~7;
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p < Kv; p += 8)
    {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    }
    float s = hsum256(acc);
    for (; p < K; ++p) s += a[p] * b[p];
    return s;

}

void matmul_optimized(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc)
{
    const int Kv = K & ~7;
    const int M4 = M & ~3;
    for (int jc = 0; jc < N; jc += JC)
    {
        const int jmax = (jc + JC < N) ? (jc + JC) : N;
        const int jmax4 = jc + ((jmax - jc) & ~3);
        int i = 0;
        for (; i < M4; i += 4)
        {
            const float* a0 = A + (long)(i + 0) * lda;
            const float* a1 = A + (long)(i + 1) * lda;
            const float* a2 = A + (long)(i + 2) * lda;
            const float* a3 = A + (long)(i + 3) * lda;

            float* c0 = C + (long)(i + 0) * ldc;
            float* c1 = C + (long)(i + 1) * ldc;
            float* c2 = C + (long)(i + 2) * ldc;
            float* c3 = C + (long)(i + 3) * ldc;

            int j=jc;
            for (; j < jmax4; j += 4)
            {
               const float* b0 = B + (long)(j + 0) * ldb;
               const float* b1 = B + (long)(j + 1) * ldb;
               const float* b2 = B + (long)(j + 2) * ldb;
               const float* b3 = B + (long)(j + 3) * ldb;

               __m256 c00=_mm256_setzero_ps(),c01=_mm256_setzero_ps(),c02=_mm256_setzero_ps(),c03=_mm256_setzero_ps();
               __m256 c10=_mm256_setzero_ps(),c11=_mm256_setzero_ps(),c12=_mm256_setzero_ps(),c13=_mm256_setzero_ps();
               __m256 c20=_mm256_setzero_ps(),c21=_mm256_setzero_ps(),c22=_mm256_setzero_ps(),c23=_mm256_setzero_ps();
               __m256 c30=_mm256_setzero_ps(),c31=_mm256_setzero_ps(),c32=_mm256_setzero_ps(),c33=_mm256_setzero_ps();
                
               int p = 0;
                for (; p < Kv; p += 8)
                {
                    #if PF_D > 0
                    _mm_prefetch((const char*)(b0+p+PF_D),PF_L);
                    _mm_prefetch((const char*)(b1+p+PF_D),PF_L);
                    _mm_prefetch((const char*)(b2+p+PF_D),PF_L);
                    _mm_prefetch((const char*)(b3+p+PF_D),PF_L);
                    #endif

                    __m256 vb0=_mm256_loadu_ps(b0+p);
                    __m256 vb1=_mm256_loadu_ps(b1+p);
                    __m256 vb2=_mm256_loadu_ps(b2+p);
                    __m256 vb3=_mm256_loadu_ps(b3+p);

                    __m256 va;
                    va =_mm256_loadu_ps(a0+p);
                    c00=_mm256_fmadd_ps(va,vb0,c00);
                    c01=_mm256_fmadd_ps(va,vb1,c01);
                    c02=_mm256_fmadd_ps(va,vb2,c02);
                    c03=_mm256_fmadd_ps(va,vb3,c03);

                    va=_mm256_loadu_ps(a1+p);
                    c10=_mm256_fmadd_ps(va,vb0,c10);
                    c11=_mm256_fmadd_ps(va,vb1,c11);
                    c12=_mm256_fmadd_ps(va,vb2,c12);
                    c13=_mm256_fmadd_ps(va,vb3,c13);

                    va=_mm256_loadu_ps(a2+p);
                    c20=_mm256_fmadd_ps(va,vb0,c20);
                    c21=_mm256_fmadd_ps(va,vb1,c21);
                    c22=_mm256_fmadd_ps(va,vb2,c22);
                    c23=_mm256_fmadd_ps(va,vb3,c23);

                    va=_mm256_loadu_ps(a3+p);
                    c30=_mm256_fmadd_ps(va,vb0,c30);
                    c31=_mm256_fmadd_ps(va,vb1,c31);
                    c32=_mm256_fmadd_ps(va,vb2,c32);
                    c33=_mm256_fmadd_ps(va,vb3,c33);
                }
                float r00=hsum256(c00),r01=hsum256(c01),r02=hsum256(c02),r03=hsum256(c03);
                float r10=hsum256(c10),r11=hsum256(c11),r12=hsum256(c12),r13=hsum256(c13);
                float r20=hsum256(c20),r21=hsum256(c21),r22=hsum256(c22),r23=hsum256(c23);
                float r30=hsum256(c30),r31=hsum256(c31),r32=hsum256(c32),r33=hsum256(c33);

                for (; p < K; ++p)
                {
                    float x0 = a0[p];
                    float x1 = a1[p];
                    float x2 = a2[p];
                    float x3 = a3[p];

                    float y0 = b0[p];
                    float y1 = b1[p];
                    float y2 = b2[p];
                    float y3 = b3[p];

                    r00+=x0*y0; r01+=x0*y1; r02+=x0*y2; r03+=x0*y3;
                    r10+=x1*y0; r11+=x1*y1; r12+=x1*y2; r13+=x1*y3;
                    r20+=x2*y0; r21+=x2*y1; r22+=x2*y2; r23+=x2*y3;
                    r30+=x3*y0; r31+=x3*y1; r32+=x3*y2; r33+=x3*y3;
                }
                c0[j]=r00; c0[j+1]=r01; c0[j+2]=r02; c0[j+3]=r03;
                c1[j]=r10; c1[j+1]=r11; c1[j+2]=r12; c1[j+3]=r13;
                c2[j]=r20; c2[j+1]=r21; c2[j+2]=r22; c2[j+3]=r23;
                c3[j]=r30; c3[j+1]=r31; c3[j+2]=r32; c3[j+3]=r33;
            }

            for (; j < jmax; ++j)
            {
                const float* b = B + (long)j * ldb;
                c0[j]=dot_scalar(a0,b,K);
                c1[j]=dot_scalar(a1,b,K);
                c2[j]=dot_scalar(a2,b,K);
                c3[j]=dot_scalar(a3,b,K);
            }
        }
        for (; i < M; ++i)
        {
            const float* a = A + (long)i * lda;
            float* c = C + (long)i * ldc;
            for (int j = jc; j < jmax; ++j)
            {
                c[j] = dot_scalar(a, B + (long)j * ldb, K);
            }
        }
    }
}