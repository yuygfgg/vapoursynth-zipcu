// -use_fast_math; aggregate must use __fdiv_rn (not `/` → div.approx).

#define FMA(a, b, c) (((a) * (b)) + (c))
#define FMS(a, b, c) (((a) * (b)) - (c))
#define FNMS(a, b, c) ((c) - ((a) * (b)))

#define FLT_MAX_ 3.402823466e+38f
#define FLT_EPS_ 1.192092896e-07f

__device__ static const int smem_stride = 32 + 1;

// An eight-lane group is enough for the original limit. A larger limit uses
// two copies of the eight spatial lanes: the first half does the transforms
// and aggregation, while the second half provides extra candidate ranks.
#if MAX_GROUP_SIZE > 8
#define GROUP_WIDTH 16
#define GROUPS_PER_WARP 2
#else
#define GROUP_WIDTH 8
#define GROUPS_PER_WARP 4
#endif

template <auto transform_impl, int stride = 256, int howmany = 8, int howmany_stride = 32>
__device__ static inline void transform_pack8_interleave4(float *__restrict__ data, float *__restrict__ buffer) {
#pragma unroll
    for (int iter = 0; iter < howmany; ++iter, data += howmany_stride) {
        float v[8];

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = data[i * stride];
        }

        transform_impl(v);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            data[i * stride] = v[i];
        }
    }
}

template <bool forward>
__device__ static inline void dct(float v[8]) {
    if constexpr (forward) {
        float KP414213562{+0.414213562373095048801688724209698078569671875};
        float KP1_847759065{+1.847759065022573512256366378793576573644833252};
        float KP198912367{+0.198912367379658006911597622644676228597850501};
        float KP1_961570560{+1.961570560806460898252364472268478073947867462};
        float KP1_414213562{+1.414213562373095048801688724209698078569671875};
        float KP668178637{+0.668178637919298919997757686523080761552472251};
        float KP1_662939224{+1.662939224605090474157576755235811513477121624};
        float KP707106781{+0.707106781186547524400844362104849039284835938};

        auto T1 = v[0];
        auto T2 = v[7];
        auto T3 = T1 - T2;
        auto Tj = T1 + T2;
        auto Tc = v[4];
        auto Td = v[3];
        auto Te = Tc - Td;
        auto Tk = Tc + Td;
        auto T4 = v[2];
        auto T5 = v[5];
        auto T6 = T4 - T5;
        auto T7 = v[1];
        auto T8 = v[6];
        auto T9 = T7 - T8;
        auto Ta = T6 + T9;
        auto Tn = T7 + T8;
        auto Tf = T6 - T9;
        auto Tm = T4 + T5;
        auto Tb = FNMS(KP707106781, Ta, T3);
        auto Tg = FNMS(KP707106781, Tf, Te);
        v[3] = KP1_662939224 * (FMA(KP668178637, Tg, Tb));
        v[5] = -(KP1_662939224 * (FNMS(KP668178637, Tb, Tg)));
        auto Tp = Tj + Tk;
        auto Tq = Tm + Tn;
        v[4] = KP1_414213562 * (Tp - Tq);
        v[0] = KP1_414213562 * (Tp + Tq);
        auto Th = FMA(KP707106781, Ta, T3);
        auto Ti = FMA(KP707106781, Tf, Te);
        v[1] = KP1_961570560 * (FNMS(KP198912367, Ti, Th));
        v[7] = KP1_961570560 * (FMA(KP198912367, Th, Ti));
        auto Tl = Tj - Tk;
        auto To = Tm - Tn;
        v[2] = KP1_847759065 * (FNMS(KP414213562, To, Tl));
        v[6] = KP1_847759065 * (FMA(KP414213562, Tl, To));
    } else {
        float KP1_662939224{+1.662939224605090474157576755235811513477121624};
        float KP668178637{+0.668178637919298919997757686523080761552472251};
        float KP1_961570560{+1.961570560806460898252364472268478073947867462};
        float KP198912367{+0.198912367379658006911597622644676228597850501};
        float KP1_847759065{+1.847759065022573512256366378793576573644833252};
        float KP707106781{+0.707106781186547524400844362104849039284835938};
        float KP414213562{+0.414213562373095048801688724209698078569671875};
        float KP1_414213562{+1.414213562373095048801688724209698078569671875};

        auto T1 = v[0] * KP1_414213562;
        auto T2 = v[4];
        auto T3 = FMA(KP1_414213562, T2, T1);
        auto Tj = FNMS(KP1_414213562, T2, T1);
        auto T4 = v[2];
        auto T5 = v[6];
        auto T6 = FMA(KP414213562, T5, T4);
        auto Tk = FMS(KP414213562, T4, T5);
        auto T8 = v[1];
        auto Td = v[7];
        auto T9 = v[5];
        auto Ta = v[3];
        auto Tb = T9 + Ta;
        auto Te = Ta - T9;
        auto Tc = FMA(KP707106781, Tb, T8);
        auto Tn = FNMS(KP707106781, Te, Td);
        auto Tf = FMA(KP707106781, Te, Td);
        auto Tm = FNMS(KP707106781, Tb, T8);
        auto T7 = FMA(KP1_847759065, T6, T3);
        auto Tg = FMA(KP198912367, Tf, Tc);
        v[7] = FNMS(KP1_961570560, Tg, T7);
        v[0] = FMA(KP1_961570560, Tg, T7);
        auto Tp = FNMS(KP1_847759065, Tk, Tj);
        auto Tq = FMA(KP668178637, Tm, Tn);
        v[5] = FNMS(KP1_662939224, Tq, Tp);
        v[2] = FMA(KP1_662939224, Tq, Tp);
        auto Th = FNMS(KP1_847759065, T6, T3);
        auto Ti = FNMS(KP198912367, Tc, Tf);
        v[3] = FNMS(KP1_961570560, Ti, Th);
        v[4] = FMA(KP1_961570560, Ti, Th);
        auto Tl = FMA(KP1_847759065, Tk, Tj);
        auto To = FNMS(KP668178637, Tn, Tm);
        v[6] = FNMS(KP1_662939224, To, Tl);
        v[1] = FMA(KP1_662939224, To, Tl);
    }
}

#if MAX_GROUP_SIZE > 8
__device__ __constant__ static const float dct16_coeffs[16][16] = {
    {1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f,
     1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f},
    {1.99036945f, 1.91388067f, 1.76384253f, 1.54602091f, 1.26878657f, 0.942793474f, 0.580569355f, 0.196034281f,
     -0.196034281f, -0.580569355f, -0.942793474f, -1.26878657f, -1.54602091f, -1.76384253f, -1.91388067f, -1.99036945f},
    {1.96157056f, 1.66293922f, 1.11114047f, 0.390180644f, -0.390180644f, -1.11114047f, -1.66293922f, -1.96157056f,
     -1.96157056f, -1.66293922f, -1.11114047f, -0.390180644f, 0.390180644f, 1.11114047f, 1.66293922f, 1.96157056f},
    {1.91388067f, 1.26878657f, 0.196034281f, -0.942793474f, -1.76384253f, -1.99036945f, -1.54602091f, -0.580569355f,
     0.580569355f, 1.54602091f, 1.99036945f, 1.76384253f, 0.942793474f, -0.196034281f, -1.26878657f, -1.91388067f},
    {1.84775907f, 0.765366865f, -0.765366865f, -1.84775907f, -1.84775907f, -0.765366865f, 0.765366865f, 1.84775907f,
     1.84775907f, 0.765366865f, -0.765366865f, -1.84775907f, -1.84775907f, -0.765366865f, 0.765366865f, 1.84775907f},
    {1.76384253f, 0.196034281f, -1.54602091f, -1.91388067f, -0.580569355f, 1.26878657f, 1.99036945f, 0.942793474f,
     -0.942793474f, -1.99036945f, -1.26878657f, 0.580569355f, 1.91388067f, 1.54602091f, -0.196034281f, -1.76384253f},
    {1.66293922f, -0.390180644f, -1.96157056f, -1.11114047f, 1.11114047f, 1.96157056f, 0.390180644f, -1.66293922f,
     -1.66293922f, 0.390180644f, 1.96157056f, 1.11114047f, -1.11114047f, -1.96157056f, -0.390180644f, 1.66293922f},
    {1.54602091f, -0.942793474f, -1.91388067f, 0.196034281f, 1.99036945f, 0.580569355f, -1.76384253f, -1.26878657f,
     1.26878657f, 1.76384253f, -0.580569355f, -1.99036945f, -0.196034281f, 1.91388067f, 0.942793474f, -1.54602091f},
    {1.41421356f, -1.41421356f, -1.41421356f, 1.41421356f, 1.41421356f, -1.41421356f, -1.41421356f, 1.41421356f,
     1.41421356f, -1.41421356f, -1.41421356f, 1.41421356f, 1.41421356f, -1.41421356f, -1.41421356f, 1.41421356f},
    {1.26878657f, -1.76384253f, -0.580569355f, 1.99036945f, -0.196034281f, -1.91388067f, 0.942793474f, 1.54602091f,
     -1.54602091f, -0.942793474f, 1.91388067f, 0.196034281f, -1.99036945f, 0.580569355f, 1.76384253f, -1.26878657f},
    {1.11114047f, -1.96157056f, 0.390180644f, 1.66293922f, -1.66293922f, -0.390180644f, 1.96157056f, -1.11114047f,
     -1.11114047f, 1.96157056f, -0.390180644f, -1.66293922f, 1.66293922f, 0.390180644f, -1.96157056f, 1.11114047f},
    {0.942793474f, -1.99036945f, 1.26878657f, 0.580569355f, -1.91388067f, 1.54602091f, 0.196034281f, -1.76384253f,
     1.76384253f, -0.196034281f, -1.54602091f, 1.91388067f, -0.580569355f, -1.26878657f, 1.99036945f, -0.942793474f},
    {0.765366865f, -1.84775907f, 1.84775907f, -0.765366865f, -0.765366865f, 1.84775907f, -1.84775907f, 0.765366865f,
     0.765366865f, -1.84775907f, 1.84775907f, -0.765366865f, -0.765366865f, 1.84775907f, -1.84775907f, 0.765366865f},
    {0.580569355f, -1.54602091f, 1.99036945f, -1.76384253f, 0.942793474f, 0.196034281f, -1.26878657f, 1.91388067f,
     -1.91388067f, 1.26878657f, -0.196034281f, -0.942793474f, 1.76384253f, -1.99036945f, 1.54602091f, -0.580569355f},
    {0.390180644f, -1.11114047f, 1.66293922f, -1.96157056f, 1.96157056f, -1.66293922f, 1.11114047f, -0.390180644f,
     -0.390180644f, 1.11114047f, -1.66293922f, 1.96157056f, -1.96157056f, 1.66293922f, -1.11114047f, 0.390180644f},
    {0.196034281f, -0.580569355f, 0.942793474f, -1.26878657f, 1.54602091f, -1.76384253f, 1.91388067f, -1.99036945f,
     1.99036945f, -1.91388067f, 1.76384253f, -1.54602091f, 1.26878657f, -0.942793474f, 0.580569355f, -0.196034281f},
};
#endif

template <bool forward>
__device__ static inline void haar(float v[8]) {
    if constexpr (forward) {
        float KP1_414213562{+1.414213562373095048801688724209698078569671875};
        float KP2_000000000{+2.000000000000000000000000000000000000000000000};

        auto T1 = v[0] + v[1];
        auto T2 = v[0] - v[1];
        auto T3 = v[2] + v[3];
        auto T4 = v[2] - v[3];
        auto T5 = v[4] + v[5];
        auto T6 = v[4] - v[5];
        auto T7 = v[6] + v[7];
        auto T8 = v[6] - v[7];

        auto T9 = T1 + T3;
        auto T10 = KP1_414213562 * (T1 - T3);
        auto T11 = T5 + T7;
        auto T12 = KP1_414213562 * (T5 - T7);

        auto scale = KP1_414213562;
        v[0] = scale * (T9 + T11);
        v[1] = scale * (T9 - T11);
        v[2] = scale * T10;
        v[3] = scale * T12;
        v[4] = scale * KP2_000000000 * T2;
        v[5] = scale * KP2_000000000 * T4;
        v[6] = scale * KP2_000000000 * T6;
        v[7] = scale * KP2_000000000 * T8;
    } else {
        // The inverse is the transpose of the scaled forward matrix.  Keep
        // all input coefficients live until every output has been formed.
        constexpr float s = 1.414213562373095048801688724209698078569671875f;
        const float c0 = v[0], c1 = v[1], c2 = v[2], c3 = v[3];
        const float c4 = v[4], c5 = v[5], c6 = v[6], c7 = v[7];
        const float dc_plus = s * (c0 + c1);
        const float dc_minus = s * (c0 - c1);
        const float two_s = 2.0f * s;

        v[0] = dc_plus + 2.0f * c2 + two_s * c4;
        v[1] = dc_plus + 2.0f * c2 - two_s * c4;
        v[2] = dc_plus - 2.0f * c2 + two_s * c5;
        v[3] = dc_plus - 2.0f * c2 - two_s * c5;
        v[4] = dc_minus + 2.0f * c3 + two_s * c6;
        v[5] = dc_minus + 2.0f * c3 - two_s * c6;
        v[6] = dc_minus - 2.0f * c3 + two_s * c7;
        v[7] = dc_minus - 2.0f * c3 - two_s * c7;
    }
}

template <bool forward>
__device__ static inline void wht(float v[8]) {
    float KP1_414213562{+1.414213562373095048801688724209698078569671875};

    auto T1 = v[0] + v[1];
    auto T2 = v[0] - v[1];
    auto T3 = v[2] + v[3];
    auto T4 = v[2] - v[3];
    auto T5 = v[4] + v[5];
    auto T6 = v[4] - v[5];
    auto T7 = v[6] + v[7];
    auto T8 = v[6] - v[7];

    auto T9 = T1 + T3;
    auto T10 = T1 - T3;
    auto T11 = T2 + T4;
    auto T12 = T2 - T4;
    auto T13 = T5 + T7;
    auto T14 = T5 - T7;
    auto T15 = T6 + T8;
    auto T16 = T6 - T8;

    float scale = KP1_414213562;
    v[0] = scale * (T9 + T13);
    v[1] = scale * (T9 - T13);
    v[2] = scale * (T10 - T14);
    v[3] = scale * (T10 + T14);
    v[4] = scale * (T12 + T16);
    v[5] = scale * (T12 - T16);
    v[6] = scale * (T11 - T15);
    v[7] = scale * (T11 + T15);
}

template <bool forward>
__device__ static inline void bior1_5(float v[8]) {
    if constexpr (forward) {
        float KP1_414213562{+1.414213562373095048801688724209698078569671875};
        float KP877670597{+0.877670597010003062405456290501163901200537360};
        float KP1_797135031{+1.797135031972863413496886690073811797696338403};
        float KP2_277437593{+2.277437593371384746027611934034934695963515886};
        float KP1_609389232{+1.609389232649111887192845766718020518480884560};
        float KP334024180{+0.334024180361136429417383083658457088741315663};
        float KP2_828427124{+2.828427124746190097603377448419396157139343751};

        auto T1 = v[0] + v[3];
        auto T2 = v[0] - v[3];
        auto T3 = v[1] + v[2];
        auto T4 = v[1] - v[2];
        auto T5 = v[4] + v[7];
        auto T6 = v[4] - v[7];
        auto T7 = v[5] + v[6];
        auto T8 = v[5] - v[6];
        auto T9 = v[0] - v[1];
        auto T10 = v[2] - v[3];
        auto T11 = v[4] - v[5];
        auto T12 = v[6] - v[7];

        v[0] = KP1_414213562 * (T1 + T5 + T3 + T7);
        v[1] = KP877670597 * (T1 - T5) + KP1_797135031 * (T3 - T7);
        v[2] = KP2_277437593 * T2 + KP1_609389232 * T4 + KP334024180 * (T8 - T6);
        v[3] = KP2_277437593 * T6 + KP1_609389232 * T8 + KP334024180 * (T4 - T2);
        v[4] = KP2_828427124 * T9;
        v[5] = KP2_828427124 * T10;
        v[6] = KP2_828427124 * T11;
        v[7] = KP2_828427124 * T12;
    } else {
        float KP1_414213562{+1.414213562373095048801688724209698078569671875};
        float KP1_495435764{+1.495435764250674860795011090214036706658653686};
        float KP2_058234225{+2.058234225009388964222454285384072231477027482};
        float KP486135912{+0.486135912065751423025580498947083714508324707};
        float KP2_828427124{+2.828427124746190097603377448419396157139343751};

        auto T1 = KP1_414213562 * v[0];
        auto T2 = KP1_495435764 * v[1];
        auto T3 = KP2_058234225 * v[2];
        auto T4 = KP2_058234225 * v[3];
        auto T5 = KP2_828427124 * v[4];
        auto T6 = KP486135912 * v[4];
        auto T7 = KP2_828427124 * v[5];
        auto T8 = KP486135912 * v[5];
        auto T9 = KP2_828427124 * v[6];
        auto T10 = KP486135912 * v[6];
        auto T11 = KP2_828427124 * v[7];
        auto T12 = KP486135912 * v[7];

        auto T13 = T1 + T2;
        auto T14 = T1 - T2;
        auto T15 = T8 - T12;
        auto T16 = T6 - T10;

        v[0] = (T13 + T3) + (T5 - T15);
        v[1] = (T13 + T3) - (T5 + T15);
        v[2] = (T13 - T3) + (T16 + T7);
        v[3] = (T13 - T3) + (T16 - T7);
        v[4] = (T14 + T4) + (T15 + T9);
        v[5] = (T14 + T4) + (T15 - T9);
        v[6] = (T14 - T4) - (T16 - T11);
        v[7] = (T14 - T4) - (T16 + T11);
    }
}

template <bool forward>
__device__ static inline void dct_n(float v[GROUP_WIDTH], int n) {
    if (n == 8) {
        dct<forward>(v);
        return;
    }
    if (n == 1) {
        v[0] *= 1.4142135623730950488f;
        return;
    }
#if MAX_GROUP_SIZE > 8
    if (n == 16) {
        float out[GROUP_WIDTH]{};
        if constexpr (forward) {
            for (int k = 0; k < 16; ++k)
#pragma unroll
                for (int j = 0; j < 16; ++j) out[k] += dct16_coeffs[k][j] * v[j];
        } else {
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int k = 0; k < 16; ++k) out[j] += dct16_coeffs[k][j] * v[k];
        }
        for (int i = 0; i < GROUP_WIDTH; ++i) v[i] = out[i];
        return;
    }
#endif
    float out[GROUP_WIDTH]{};
    constexpr float pi = 3.14159265358979323846f;
    constexpr float sqrt2 = 1.4142135623730950488f;
    if constexpr (forward) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j)
                out[k] += 2.0f * v[j] * cosf(pi * (j + 0.5f) * k / n);
            if (k == 0) out[k] *= 0.7071067811865475244f;
        }
    } else {
        const float dc = v[0] * sqrt2;
        for (int j = 0; j < n; ++j) {
            out[j] = dc;
            for (int k = 1; k < n; ++k)
                out[j] += 2.0f * v[k] * cosf(pi * (j + 0.5f) * k / n);
        }
    }
    for (int i = 0; i < GROUP_WIDTH; ++i) v[i] = i < n ? out[i] : 0.0f;
}

template <bool forward>
__device__ static inline void wht_n(float v[GROUP_WIDTH], int n) {
    if (n == 8) {
        wht<forward>(v);
        return;
    }
    if (n == 1) {
        v[0] *= 1.4142135623730950488f;
        return;
    }
    float out[GROUP_WIDTH]{};
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            const int bits = k & j;
            int parity = 0;
            for (int b = bits; b; b &= b - 1) parity ^= 1;
            out[k] += (parity ? -1.0f : 1.0f) * v[j];
        }
        out[k] *= 1.4142135623730950488f;
    }
    for (int i = 0; i < GROUP_WIDTH; ++i) v[i] = i < n ? out[i] : 0.0f;
}

template <bool forward>
__device__ static inline void haar_n(float v[GROUP_WIDTH], int n) {
    if (n == 8) {
        haar<forward>(v);
        return;
    }
    if constexpr (!forward) {
        float out[GROUP_WIDTH]{};
#pragma unroll
        for (int j = 0; j < n; ++j) {
#pragma unroll
            for (int k = 0; k < n; ++k) {
                float basis;
                if (k == 0) {
                    basis = 1.4142135623730950488f;
                } else {
                    const int level = 31 - __clz(static_cast<unsigned>(k));
                    const int span = n >> level;
                    const int block = k - (1 << level);
                    const int begin = block * span;
                    basis = (j >= begin && j < begin + span) ?
                        ((j - begin) < (span >> 1) ? 1.0f : -1.0f) * sqrtf(2.0f * n / span) : 0.0f;
                }
                out[j] += basis * v[k];
            }
        }
#pragma unroll
        for (int i = 0; i < GROUP_WIDTH; ++i) v[i] = i < n ? out[i] : 0.0f;
        return;
    }
    if (n == 1) {
        v[0] *= 1.4142135623730950488f;
        return;
    }
    constexpr float s = 1.4142135623730950488f;
    if (n == 2) {
        const float a = v[0], b = v[1];
        v[0] = s * (a + b);
        v[1] = s * (a - b);
        return;
    }
    const float a = v[0], b = v[1], c = v[2], d = v[3];
    if constexpr (forward) {
        v[0] = s * (a + b + c + d);
        v[1] = s * (a + b - c - d);
        v[2] = 2.0f * (a - b);
        v[3] = 2.0f * (c - d);
    } else {
        v[0] = s * (a + b) + 2.0f * c;
        v[1] = s * (a + b) - 2.0f * c;
        v[2] = s * (a - b) + 2.0f * d;
        v[3] = s * (a - b) - 2.0f * d;
    }
    for (int i = 4; i < GROUP_WIDTH; ++i) v[i] = 0.0f;
}

template <bool forward>
__device__ static inline void bior1_5_n(float v[GROUP_WIDTH], int n) {
    if (n == 8) bior1_5<forward>(v);
    else haar_n<forward>(v, n);
}

#define BM3D_CAT_I(a, b) a##b
#define BM3D_CAT(a, b) BM3D_CAT_I(a, b)
#define BM3D_GROUP_TRANSFORM(name, forward, v, n) BM3D_CAT(name, _n)<forward>(v, n)

template <bool forward>
__device__ static inline void transform_group(float *data, int group_size) {
#pragma unroll
    for (int iter = 0; iter < 8; ++iter, ++data) {
        float v[GROUP_WIDTH]{};
#pragma unroll
        for (int i = 0; i < MAX_GROUP_SIZE; ++i) v[i] = data[i * 8];
        BM3D_GROUP_TRANSFORM(TRANSFORM_1D, forward, v, group_size);
#pragma unroll
        for (int i = 0; i < MAX_GROUP_SIZE; ++i) data[i * 8] = v[i];
    }
}

__device__ static inline float reduce_subwarp(float x, unsigned int mask) {
    x += __shfl_xor_sync(mask, x, 1, 8);
    x += __shfl_xor_sync(mask, x, 2, 8);
    x += __shfl_xor_sync(mask, x, 4, 8);

    return x;
}

__device__ static inline float ssd(const float center[__restrict__ 8], const float neighbor[__restrict__ 8], unsigned int mask) {
    float errors[2]{0.0f};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = center[i] - neighbor[i];
        errors[i % 2] += val * val;
    }

    float error = errors[0] + errors[1];

    return reduce_subwarp(error, mask);
}

__device__ static inline float sad(const float center[__restrict__ 8], const float neighbor[__restrict__ 8], unsigned int mask) {
    float errors[2]{0.0f};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = center[i] - neighbor[i];
        errors[i % 2] += fabsf(val);
    }

    float error = errors[0] + errors[1];

    return reduce_subwarp(error, mask);
}

__device__ static inline float zssd(const float center[__restrict__ 8], const float neighbor[__restrict__ 8], unsigned int mask) {
    float center_sum = (((center[0] + center[1]) + (center[2] + center[3])) +
                        ((center[4] + center[5]) + (center[6] + center[7])));
    float center_mean = reduce_subwarp(center_sum, mask) * (1.0f / 64.f);

    float neighbor_sum = (((neighbor[0] + neighbor[1]) + (neighbor[2] + neighbor[3])) +
                          ((neighbor[4] + neighbor[5]) + (neighbor[6] + neighbor[7])));
    float neighbor_mean = reduce_subwarp(neighbor_sum, mask) * (1.0f / 64.f);

    float errors[2]{0.0f};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = center[i] - neighbor[i] - (center_mean - neighbor_mean);
        errors[i % 2] += val * val;
    }

    float error = errors[0] + errors[1];

    return reduce_subwarp(error, mask);
}

__device__ static inline float zsad(const float center[__restrict__ 8], const float neighbor[__restrict__ 8], unsigned int mask) {
    float center_sum = (((center[0] + center[1]) + (center[2] + center[3])) +
                        ((center[4] + center[5]) + (center[6] + center[7])));
    float center_mean = reduce_subwarp(center_sum, mask) * (1.0f / 64.f);

    float neighbor_sum = (((neighbor[0] + neighbor[1]) + (neighbor[2] + neighbor[3])) +
                          ((neighbor[4] + neighbor[5]) + (neighbor[6] + neighbor[7])));
    float neighbor_mean = reduce_subwarp(neighbor_sum, mask) * (1.0f / 64.f);

    float errors[2]{0.0f};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = center[i] - neighbor[i] - (center_mean - neighbor_mean);
        errors[i % 2] += fabsf(val);
    }

    float error = errors[0] + errors[1];

    return reduce_subwarp(error, mask);
}

__device__ static inline float ssd_norm(const float center[__restrict__ 8], const float neighbor[__restrict__ 8], unsigned int mask) {
    float center_ssds[2]{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        center_ssds[i % 2] += center[i] * center[i];
    }
    float center_ssd = center_ssds[0] + center_ssds[1];
    float center_norm = sqrtf(reduce_subwarp(center_ssd, mask));

    float neighbor_ssds[2]{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        neighbor_ssds[i % 2] += neighbor[i] * neighbor[i];
    }
    float neighbor_ssd = neighbor_ssds[0] + neighbor_ssds[1];
    float neighbor_norm = sqrtf(reduce_subwarp(neighbor_ssd, mask));

    float errors[2]{0.0f};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = center[i] * neighbor[i];
        errors[i % 2] += val;
    }

    float error = errors[0] + errors[1];

    return 2.0f - 2.0f * reduce_subwarp(error, mask) / (center_norm * neighbor_norm + FLT_EPS_);
}

template <int stride = 256, int howmany = 8, int howmany_stride = 32>
__device__ static inline void transpose_pack8_interleave4(
    float *__restrict__ data, float *__restrict__ buffer, unsigned int mask) {
    int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));

#pragma unroll
    for (int iter = 0; iter < howmany; ++iter, data += howmany_stride) {
        __syncwarp(mask);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            buffer[i * smem_stride + lane_id] = data[i * stride];
        }

        __syncwarp(mask);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            data[i * stride] = buffer[(lane_id % 8) * smem_stride + (lane_id & -8) + i];
        }
    }
}

// Arch-gated k reduction (sm_75/86 vs sequential): hard-thr vs Wiener shapes differ.
template <int stride = 32>
__device__ static inline float hard_thresholding(float *data, float sigma, int group_size, unsigned int mask) {
#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
    float ks[4]{};
#else
    float k{};
#endif

#pragma unroll
    for (int i = 0; i < MAX_GROUP_SIZE * 8; ++i) {
        auto val = data[i * stride];

        // The 8-point spatial transforms are normalized. After normalizing
        // short group transforms as well, only the group dimension changes
        // the coefficient noise gain: sqrt(group_size / 8).
        const int z = i >> 3;
        const float thr = sigma * sqrtf(group_size * (1.0f / 8.0f));

        float flag = (z < group_size) && fabsf(val) >= thr;

#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
        ks[i % 4] += flag;
#else
        k += flag;
#endif
        data[i * stride] = flag ? (val * __fdiv_rn(1.0f, 512.0f * group_size)) : 0.0f;
    }

#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
    float k{(ks[0] + ks[1]) + (ks[2] + ks[3])};
#endif

    k = reduce_subwarp(k, mask);

    return 1.0f / fmaxf(k, 1.0f);
}

__device__ static inline float collaborative_hard(float *__restrict__ denoising_patch, float sigma, float *__restrict__ buffer, int group_size, unsigned int mask) {
    constexpr int stride1 = 1;
    constexpr int stride2 = stride1 * 8;

#pragma unroll
    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<TRANSFORM_2D<true>, stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer);
        transpose_pack8_interleave4<stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer, mask);
    }
    transform_group<true>(denoising_patch, group_size);

    float adaptive_weight = hard_thresholding<stride1>(denoising_patch, sigma, group_size, mask);

#pragma unroll
    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<TRANSFORM_2D<false>, stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer);
        transpose_pack8_interleave4<stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer, mask);
    }
    transform_group<false>(denoising_patch, group_size);

    return adaptive_weight;
}

template <int stride = 32>
__device__ static inline float wiener_filtering(float *__restrict__ data, float *__restrict__ ref, float sigma, int group_size, unsigned int mask) {
#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
    float ks[4]{};
#else
    float k{};
#endif

#pragma unroll
    for (int i = 0; i < MAX_GROUP_SIZE * 8; ++i) {
        auto val = data[i * stride];
        auto ref_val = ref[i * stride];
        const float scaled_sigma = sigma * sqrtf(group_size * (1.0f / 8.0f));
        float coeff = (ref_val * ref_val) / (ref_val * ref_val + scaled_sigma * scaled_sigma);
        if ((i >> 3) >= group_size) coeff = 0.0f;
        val *= coeff;
#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
        ks[i % 4] += coeff * coeff;
#else
        k += coeff * coeff;
#endif
        data[i * stride] = val * __fdiv_rn(1.0f, 512.0f * group_size);
    }

#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
    float k{(ks[0] + ks[1]) + (ks[2] + ks[3])};
#endif

    k = reduce_subwarp(k, mask);

    return 1.0f / fmaxf(k, FLT_EPS_);
}

__device__ static inline float collaborative_wiener(
    float *__restrict__ denoising_patch, float *__restrict__ ref_patch, float sigma, float *__restrict__ buffer, int group_size, unsigned int mask) {
    constexpr int stride1 = 1;
    constexpr int stride2 = stride1 * 8;

#pragma unroll
    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<TRANSFORM_2D<true>, stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer);
        transpose_pack8_interleave4<stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer, mask);
    }
    transform_group<true>(denoising_patch, group_size);

#pragma unroll
    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<TRANSFORM_2D<true>, stride1, MAX_GROUP_SIZE, stride2>(ref_patch, buffer);
        transpose_pack8_interleave4<stride1, MAX_GROUP_SIZE, stride2>(ref_patch, buffer, mask);
    }
    transform_group<true>(ref_patch, group_size);

    float adaptive_weight = wiener_filtering<stride1>(denoising_patch, ref_patch, sigma, group_size, mask);

#pragma unroll
    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<TRANSFORM_2D<false>, stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer);
        transpose_pack8_interleave4<stride1, MAX_GROUP_SIZE, stride2>(denoising_patch, buffer, mask);
    }
    transform_group<false>(denoising_patch, group_size);

    return adaptive_weight;
}

#if TEMPORAL
#define KRADIUS RADIUS
#else
#define KRADIUS 0
#endif

#define TEMPORAL_WIDTH (2 * KRADIUS + 1)
#define TEMPORAL_STRIDE (HEIGHT * STRIDE)
#define PLANE_STRIDE (TEMPORAL_WIDTH * TEMPORAL_STRIDE)
#define NUM_PLANES (CHROMA ? 3 : 1)
#define CLIP_STRIDE (NUM_PLANES * TEMPORAL_WIDTH * TEMPORAL_STRIDE)

#define THREADS (32 * WARPS)
#if WARPS == 1
#define MINB 16
#elif WARPS == 2
#define MINB 10
#elif WARPS == 4
#define MINB 5
#else
#define MINB 2
#endif

extern "C" __global__ __launch_bounds__(THREADS, MINB) void bm3d(
    /* shape: [NUM_PLANES, TEMPORAL_WIDTH, 2, HEIGHT, STRIDE] */
    float *__restrict__ res,
    /* shape: [(FINAL ? 2 : 1), NUM_PLANES, TEMPORAL_WIDTH, HEIGHT, STRIDE] */
    const float *__restrict__ src) {

    __shared__ float buffer_all[WARPS][8 * smem_stride];
    __shared__ int match_x[WARPS][GROUPS_PER_WARP][GROUP_WIDTH];
    __shared__ int match_y[WARPS][GROUPS_PER_WARP][GROUP_WIDTH];
#if TEMPORAL
    __shared__ int match_z[WARPS][GROUPS_PER_WARP][GROUP_WIDTH];
#endif

    int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));

    const int warp_id = threadIdx.x >> 5;
    float *const buffer = buffer_all[warp_id];

    const int gid = blockIdx.x * WARPS + warp_id;

    const int group_lane = lane_id & (GROUP_WIDTH - 1);
    const int group_id = lane_id / GROUP_WIDTH;
    const unsigned int group_mask = ((1u << GROUP_WIDTH) - 1u) << (lane_id & -GROUP_WIDTH);
#if GROUP_WIDTH == 8
    const unsigned int half_mask = group_mask;
#else
    const unsigned int half_mask = 0xFFu << (lane_id & -8);
#endif
    const int sub_lane_id = lane_id & 7;
    int x = (GROUPS_PER_WARP * gid + group_id) * BLOCK_STEP;
    int y = BLOCK_STEP * blockIdx.y;
    if (x >= WIDTH - 8 + BLOCK_STEP || y >= HEIGHT - 8 + BLOCK_STEP) {
        return;
    }

    x = min(x, WIDTH - 8);
    y = min(y, HEIGHT - 8);

    float current_patch[8];
    const float *const srcpc = &src[KRADIUS * TEMPORAL_STRIDE + sub_lane_id];

    int membermask = group_mask;
    float errors16 = FLT_MAX_;
    int index16_x = 0;
    int index16_y = 0;

    if constexpr (TAU_MATCH > 0.0f) {
    {
        const float *srcp = &srcpc[y * STRIDE + x];

        if (group_lane < 8) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                current_patch[i] = srcp[i * STRIDE];
            }
        }
    }

    {
        int left = max(x - BM_RANGE, 0);
        int right = min(x + BM_RANGE, WIDTH - 8);
        int top = max(y - BM_RANGE, 0);
        int bottom = min(y + BM_RANGE, HEIGHT - 8);

        const float *srcp_row = srcpc + (top * STRIDE + left);
        for (int row_i = top; row_i <= bottom; ++row_i) {
            const float *srcp_col = srcp_row;
            for (int col_i = left; col_i <= right; ++col_i) {
                auto active_mask = membermask;

                __syncwarp(membermask);

                float error = 0.0f;
                if (group_lane < 8) {
                    float neighbor_patch[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        neighbor_patch[i] = srcp_col[i * STRIDE];
                    }
                    error = BM_ERROR(current_patch, neighbor_patch, half_mask);
                }
                error = __shfl_sync(group_mask, error, 0, GROUP_WIDTH);

                auto pre_error = __shfl_up_sync(active_mask, errors16, 1, GROUP_WIDTH);
                int pre_index_x = __shfl_up_sync(active_mask, index16_x, 1, GROUP_WIDTH);
                int pre_index_y = __shfl_up_sync(active_mask, index16_y, 1, GROUP_WIDTH);

                int flag = (col_i != x || row_i != y) && error <= TAU_MATCH && error < errors16;
                int pre_flag = __shfl_up_sync(active_mask, flag, 1, GROUP_WIDTH);

                if (flag) {
                    int first = (group_lane == 0) || (!pre_flag);
                    errors16 = first ? error : pre_error;
                    index16_x = first ? col_i : pre_index_x;
                    index16_y = first ? row_i : pre_index_y;
                }

                ++srcp_col;
            }

            srcp_row += STRIDE;
        }
    }
    }
    [[maybe_unused]] int index16_z = KRADIUS;

#if TEMPORAL
    if constexpr (TAU_MATCH > 0.0f) {
    {
        membermask = group_mask;

        int center_index16_x = index16_x;
        int center_index16_y = index16_y;

#pragma unroll
        for (int direction = -1; direction <= 1; direction += 2) {
            int last_index16_x = center_index16_x;
            int last_index16_y = center_index16_y;

            for (int t = 1; t <= KRADIUS; ++t) {
                int temporal_index = KRADIUS + direction * t;
                float frame_errors16 = FLT_MAX_;
                int frame_index16_x = 0;
                int frame_index16_y = 0;

                const float *temporal_srcpc = &src[temporal_index * TEMPORAL_STRIDE + sub_lane_id];

                for (int i = 0; i < PS_NUM; ++i) {
                    int xx = __shfl_sync(group_mask, last_index16_x, i, GROUP_WIDTH);
                    int yy = __shfl_sync(group_mask, last_index16_y, i, GROUP_WIDTH);

                    int left = max(xx - PS_RANGE, 0);
                    int right = min(xx + PS_RANGE, WIDTH - 8);
                    int top = max(yy - PS_RANGE, 0);
                    int bottom = min(yy + PS_RANGE, HEIGHT - 8);

                    const float *srcp_row = &temporal_srcpc[top * STRIDE + left];
                    for (int row_i = top; row_i <= bottom; ++row_i) {
                        const float *srcp_col = srcp_row;
                        for (int col_i = left; col_i <= right; ++col_i) {
                            auto active_mask = membermask;

                            __syncwarp(membermask);

                            float error = 0.0f;
                            if (group_lane < 8) {
                                float neighbor_patch[8];
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    neighbor_patch[i] = srcp_col[i * STRIDE];
                                }
                                error = BM_ERROR(current_patch, neighbor_patch, half_mask);
                            }
                            error = __shfl_sync(group_mask, error, 0, GROUP_WIDTH);

                            float pre_error = __shfl_up_sync(active_mask, frame_errors16, 1, GROUP_WIDTH);
                            int pre_index_x = __shfl_up_sync(active_mask, frame_index16_x, 1, GROUP_WIDTH);
                            int pre_index_y = __shfl_up_sync(active_mask, frame_index16_y, 1, GROUP_WIDTH);

                            int flag = error <= TAU_MATCH && error < frame_errors16;
                            int pre_flag = __shfl_up_sync(active_mask, flag, 1, GROUP_WIDTH);

                            if (flag) {
                                int first = (group_lane == 0) || (!pre_flag);
                                frame_errors16 = first ? error : pre_error;
                                frame_index16_x = first ? col_i : pre_index_x;
                                frame_index16_y = first ? row_i : pre_index_y;
                            }

                            ++srcp_col;
                        }

                        srcp_row += STRIDE;
                    }
                }

                for (int i = 0; i < PS_NUM; ++i) {
                    float tmp_error = __shfl_sync(group_mask, frame_errors16, i, GROUP_WIDTH);
                    int tmp_x = __shfl_sync(group_mask, frame_index16_x, i, GROUP_WIDTH);
                    int tmp_y = __shfl_sync(group_mask, frame_index16_y, i, GROUP_WIDTH);

                    int flag = tmp_error < errors16;
                    int pre_flag = __shfl_up_sync(group_mask, flag, 1, GROUP_WIDTH);
                    float pre_error = __shfl_up_sync(group_mask, errors16, 1, GROUP_WIDTH);
                    int pre_index_x = __shfl_up_sync(group_mask, index16_x, 1, GROUP_WIDTH);
                    int pre_index_y = __shfl_up_sync(group_mask, index16_y, 1, GROUP_WIDTH);
                    int pre_index_z = __shfl_up_sync(group_mask, index16_z, 1, GROUP_WIDTH);

                    if (flag) {
                        int first = (group_lane == 0) || (!pre_flag);
                        errors16 = first ? tmp_error : pre_error;
                        index16_x = first ? tmp_x : pre_index_x;
                        index16_y = first ? tmp_y : pre_index_y;
                        index16_z = first ? temporal_index : pre_index_z;
                    }
                }

                last_index16_x = frame_index16_x;
                last_index16_y = frame_index16_y;
            }
        }
    }
    }
#endif // TEMPORAL

    {
        if constexpr (TAU_MATCH > 0.0f) {
            const unsigned int active_mask = group_mask;

            int flag;
#if TEMPORAL
            flag = index16_x == x && index16_y == y && index16_z == KRADIUS;
#else
            flag = index16_x == x && index16_y == y;
#endif

            flag += __shfl_xor_sync(active_mask, flag, 1, GROUP_WIDTH);
            flag += __shfl_xor_sync(active_mask, flag, 2, GROUP_WIDTH);
            flag += __shfl_xor_sync(active_mask, flag, 4, GROUP_WIDTH);
#if GROUP_WIDTH > 8
            flag += __shfl_xor_sync(active_mask, flag, 8, GROUP_WIDTH);
#endif

            float pre_error = __shfl_up_sync(active_mask, errors16, 1, GROUP_WIDTH);
            int pre_index_x = __shfl_up_sync(active_mask, index16_x, 1, GROUP_WIDTH);
            int pre_index_y = __shfl_up_sync(active_mask, index16_y, 1, GROUP_WIDTH);
            [[maybe_unused]] int pre_index_z;
#if TEMPORAL
            pre_index_z = __shfl_up_sync(active_mask, index16_z, 1, GROUP_WIDTH);
#endif
            if (!flag) {
                int first = (group_lane == 0);
                errors16 = first ? 0.0f : pre_error;
                index16_x = first ? x : pre_index_x;
                index16_y = first ? y : pre_index_y;
#if TEMPORAL
                index16_z = first ? KRADIUS : pre_index_z;
#endif
            }
        } else {
            // A zero threshold skips candidate search. Keep the reference block
            // explicitly so the compacted group still has one valid member.
            errors16 = group_lane == 0 ? 0.0f : FLT_MAX_;
            index16_x = x;
            index16_y = y;
#if TEMPORAL
            index16_z = KRADIUS;
#endif
        }
    }

    // Keep every candidate that passed tau_match, up to the configured cap.
    // The reference block inserted above guarantees K>=1.
    const int matched = __popc(__ballot_sync(group_mask, errors16 <= TAU_MATCH));
    const int group_size = min(matched, MAX_GROUP_SIZE);

    match_x[warp_id][group_id][group_lane] = index16_x;
    match_y[warp_id][group_id][group_lane] = index16_y;
#if TEMPORAL
    match_z[warp_id][group_id][group_lane] = index16_z;
#endif
    __syncwarp(group_mask);

    // The second half of the group was only needed to hold candidate ranks.
    // Keeping it out of the transforms avoids duplicating all 3D work.
    if (group_lane >= 8) return;

    const unsigned int subwarp_mask = half_mask;

    float denoising_patch[MAX_GROUP_SIZE * 8];
    [[maybe_unused]] float ref_patch[MAX_GROUP_SIZE * 8];

#pragma unroll
    for (int plane = 0; plane < NUM_PLANES; ++plane) {
        float sigma;
        if (plane == 0) {
            sigma = SIGMA_Y;
        } else if (plane == 1) {
            sigma = SIGMA_U;
        } else {
            sigma = SIGMA_V;
        }

#if CHROMA
        if (sigma < FLT_EPS_) {
            src += PLANE_STRIDE;
            res += PLANE_STRIDE * 2;
            continue;
        }
#endif

        float adaptive_weight;
#if FINAL
        {
#pragma unroll
            for (int i = 0; i < MAX_GROUP_SIZE; ++i) {
                int tmp_x = match_x[warp_id][group_id][i];
                int tmp_y = match_y[warp_id][group_id][i];
                const float *refp;
#if TEMPORAL
                int tmp_z = match_z[warp_id][group_id][i];
                refp = &src[tmp_z * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x + sub_lane_id];
#else
                refp = &src[tmp_y * STRIDE + tmp_x + sub_lane_id];
#endif
                const float *srcp = &refp[CLIP_STRIDE];

#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    ref_patch[i * 8 + j] = i < group_size ? refp[j * STRIDE] : 0.0f;
                    denoising_patch[i * 8 + j] = i < group_size ? srcp[j * STRIDE] : 0.0f;
                }
            }

            adaptive_weight = collaborative_wiener(denoising_patch, ref_patch, sigma, buffer, group_size, subwarp_mask);
        }
#else
        {
#pragma unroll
            for (int i = 0; i < MAX_GROUP_SIZE; ++i) {
                int tmp_x = match_x[warp_id][group_id][i];
                int tmp_y = match_y[warp_id][group_id][i];
                const float *srcp;
#if TEMPORAL
                int tmp_z = match_z[warp_id][group_id][i];
                srcp = &src[tmp_z * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x + sub_lane_id];
#else
                srcp = &src[tmp_y * STRIDE + tmp_x + sub_lane_id];
#endif

#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    denoising_patch[i * 8 + j] = i < group_size ? srcp[j * STRIDE] : 0.0f;
                }
            }

            adaptive_weight = collaborative_hard(denoising_patch, sigma, buffer, group_size, subwarp_mask);
        }
#endif

        float *const wdstpc = &res[sub_lane_id];
        float *const weightpc = &res[TEMPORAL_STRIDE + sub_lane_id];

#pragma unroll
        for (int i = 0; i < MAX_GROUP_SIZE; ++i) {
            if (i >= group_size) continue;
            int tmp_x = match_x[warp_id][group_id][i];
            int tmp_y = match_y[warp_id][group_id][i];
            int offset;
#if TEMPORAL
            int tmp_z = match_z[warp_id][group_id][i];
            offset = tmp_z * 2 * TEMPORAL_STRIDE + tmp_y * STRIDE + tmp_x;
#else
            offset = tmp_y * STRIDE + tmp_x;
#endif

            float *wdstp = &wdstpc[offset];
            float *weightp = &weightpc[offset];

#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wdst_val = adaptive_weight * denoising_patch[i * 8 + j];
                float weight_val = adaptive_weight;

                wdst_val = (wdst_val + EXTRACTOR) - EXTRACTOR;
                weight_val = (weight_val + EXTRACTOR) - EXTRACTOR;

                atomicAdd(&wdstp[j * STRIDE], wdst_val);
                atomicAdd(&weightp[j * STRIDE], weight_val);
            }
        }

        src += PLANE_STRIDE;
        res += PLANE_STRIDE * 2;
    }
}

// __fdiv_rn required under -use_fast_math (`/` → div.approx).
extern "C" __global__ __launch_bounds__(256) void aggregate(
    /* [NUM_PLANES, HEIGHT, STRIDE] */
    float *__restrict__ dst,
    /* [NUM_PLANES, 2, HEIGHT, STRIDE] */
    const float *__restrict__ res) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    // Skipped sigma planes: do not write 0/0 NaN.
    const int plane = blockIdx.z;
    if (!((PROC_MASK >> plane) & 1)) {
        return;
    }

    const float *wdst = &res[plane * 2 * TEMPORAL_STRIDE];
    const float *weight = &wdst[TEMPORAL_STRIDE];
    float *dstp = &dst[plane * TEMPORAL_STRIDE];

    const int i = y * STRIDE + x;
    dstp[i] = weight[i] > FLT_EPS_ ? __fdiv_rn(wdst[i], weight[i]) : 0.0f;
}
