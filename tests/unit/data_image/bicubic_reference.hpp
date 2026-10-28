// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TESTS_BICUBIC_REFERENCE_HPP
#define JYPPX_LIUDING_TESTS_BICUBIC_REFERENCE_HPP
#include <array>
// Exact rational Catmull-Rom reference, a=-0.5, scaled support on
// downsampling, half-pixel centers and normalized clipped edges.
namespace bicubic_reference {
inline constexpr std::array<double, 6> kUpInput{0, 0, 1, 1, 0, 1};
inline constexpr std::array<double, 28> kUpExpected{-0.096719457013574664,
                                                    -0.088235294117647065,
                                                    -0.1053446309271631,
                                                    0,
                                                    0.44630927163109013,
                                                    0.90655043180707184,
                                                    1.0961538461538463,
                                                    0.22706043956043956,
                                                    0.17932330827067669,
                                                    0.024752671151563117,
                                                    0,
                                                    0.42667194301543332,
                                                    0.90409576573011474,
                                                    1.0961538461538463,
                                                    0.86909340659340661,
                                                    0.70987336762960029,
                                                    0.28272655322516821,
                                                    0,
                                                    0.38773248911753067,
                                                    0.89922833399287694,
                                                    1.0961538461538463,
                                                    1.1928733031674208,
                                                    0.97743197001792403,
                                                    0.41282385530389443,
                                                    0,
                                                    0.36809516050187385,
                                                    0.89677366791591984,
                                                    1.0961538461538463};
inline constexpr std::array<double, 63> kDownInput{
    0,  17, 34, 51, 68, 14, 31, 48, 65, 31, 51, 0,  20, 40, 60, 9,
    29, 49, 62, 14, 37, 60, 12, 35, 58, 10, 33, 22, 48, 3,  29, 55,
    10, 36, 62, 17, 53, 11, 40, 69, 27, 56, 14, 43, 1,  13, 45, 6,
    38, 70, 31, 63, 24, 56, 44, 8,  43, 7,  42, 6,  41, 5,  40};
inline constexpr std::array<double, 12> kDownExpected{
    26.081337154547469, 34.719449785116488, 35.942096676906161,
    39.073050547658994, 32.647899714141126, 35.918166220586485,
    36.051843667654573, 31.257964566600485, 26.895633995747463,
    34.435812367901974, 37.842365250892001, 31.832638207392868};
inline constexpr std::array<double, 2> kStepInput{0, 65535};
inline constexpr std::array<double, 5> kStepExpected{
    -6491.6745283018872, 4295.8349282296649, 32767.5, 61239.165071770338,
    72026.674528301883};
}  // namespace bicubic_reference
#endif
