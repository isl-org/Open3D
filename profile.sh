# FixedRadiusSearch
nsys profile -o radius_search \
	--cuda-memory-usage=true \
	--stats=true \
	./build/bin/examples/BenchmarkFixedRadiusSearch examples/test_data/ICP/cloud_bin_0.pcd examples/test_data/ICP/cloud_bin_1.pcd sort

# HybridSearch
nsys profile -o hybrid_search \
	--cuda-memory-usage=true \
	--stats=true \
	./build/bin/examples/BenchmarkHybridSearch examples/test_data/ICP/cloud_bin_0.pcd examples/test_data/ICP/cloud_bin_1.pcd
