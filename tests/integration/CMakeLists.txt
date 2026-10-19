# Integration tests exercise worker pools and explicit pipeline epochs.

jyppx_add_gtest(jyppx_test_large_dataset test_large_dataset.cpp)
jyppx_add_gtest(jyppx_test_concurrent_stress test_concurrent_stress.cpp)
