/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#ifndef FLANN_AUTOTUNED_INDEX_H_
#define FLANN_AUTOTUNED_INDEX_H_

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/nn/ground_truth.h"
#include "flann/nn/index_testing.h"
#include "flann/util/sampling.h"
#include "flann/algorithms/kdtree_index.h"
#include "flann/algorithms/kdtree_single_index.h"
#include "flann/algorithms/kmeans_index.h"
#include "flann/algorithms/composite_index.h"
#include "flann/algorithms/linear_index.h"
#include "flann/util/logger.h"


namespace flann
{

template<typename Distance>
inline NNIndex<Distance>*
  create_index_by_type(const flann_algorithm_t index_type,
        const Matrix<typename Distance::ElementType>& dataset, const IndexParams& params, const Distance& distance = Distance());


struct AutotunedIndexParams : public IndexParams
{
    AutotunedIndexParams(float target_precision = 0.8, float build_weight = 0.01, float memory_weight = 0, float sample_fraction = 0.1)
    {
        (*this)["algorithm"] = FLANN_INDEX_AUTOTUNED;
        // precision desired (used for autotuning, -1 otherwise)
        (*this)["target_precision"] = target_precision;
        // build tree time weighting factor
        (*this)["build_weight"] = build_weight;
        // index memory weighting factor
        (*this)["memory_weight"] = memory_weight;
        // what fraction of the dataset to use for autotuning
        (*this)["sample_fraction"] = sample_fraction;
    }
};


template <typename Distance>
class AutotunedIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    
    typedef NNIndex<Distance> BaseClass;

    typedef AutotunedIndex<Distance> IndexType;

    typedef bool needs_kdtree_distance;

    AutotunedIndex(const Matrix<ElementType>& inputData, const IndexParams& params = AutotunedIndexParams(), Distance d = Distance()) :
        BaseClass(params, d), bestIndex_(NULL), speedup_(0), dataset_(inputData)
    {
        target_precision_ = get_param(params, "target_precision",0.8f);
        build_weight_ =  get_param(params,"build_weight", 0.01f);
        memory_weight_ = get_param(params, "memory_weight", 0.0f);
        sample_fraction_ = get_param(params,"sample_fraction", 0.1f);
    }

    AutotunedIndex(const IndexParams& params = AutotunedIndexParams(), Distance d = Distance()) :
        BaseClass(params, d), bestIndex_(NULL), speedup_(0)
    {
        target_precision_ = get_param(params, "target_precision",0.8f);
        build_weight_ =  get_param(params,"build_weight", 0.01f);
        memory_weight_ = get_param(params, "memory_weight", 0.0f);
        sample_fraction_ = get_param(params,"sample_fraction", 0.1f);
    }

    AutotunedIndex(const AutotunedIndex& other) : BaseClass(other),
    		bestParams_(other.bestParams_),
    		bestSearchParams_(other.bestSearchParams_),
    		speedup_(other.speedup_),
    		dataset_(other.dataset_),
    		target_precision_(other.target_precision_),
    		build_weight_(other.build_weight_),
    		memory_weight_(other.memory_weight_),
    		sample_fraction_(other.sample_fraction_)
    {
    		bestIndex_ = other.bestIndex_->clone();
    }

    AutotunedIndex& operator=(AutotunedIndex other)
    {
    	this->swap(other);
    	return * this;
    }

    virtual ~AutotunedIndex()
    {
    	delete bestIndex_;
    }

    BaseClass* clone() const
    {
    	return new AutotunedIndex(*this);
    }

    /**
     *          Method responsible with building the index.
     */
    void buildIndex()
    {
        bestParams_ = estimateBuildParams();
        Logger::info("----------------------------------------------------\n");
        Logger::info("Autotuned parameters:\n");
        if (Logger::getLevel()>=FLANN_LOG_INFO)
        	print_params(bestParams_);
        Logger::info("----------------------------------------------------\n");

        flann_algorithm_t index_type = get_param<flann_algorithm_t>(bestParams_,"algorithm");
        bestIndex_ = create_index_by_type(index_type, dataset_, bestParams_, distance_);
        bestIndex_->buildIndex();
        speedup_ = estimateSearchParams(bestSearchParams_);
        Logger::info("----------------------------------------------------\n");
        Logger::info("Search parameters:\n");
        if (Logger::getLevel()>=FLANN_LOG_INFO)
        	print_params(bestSearchParams_);
        Logger::info("----------------------------------------------------\n");
        bestParams_["search_params"] = bestSearchParams_;
        bestParams_["speedup"] = speedup_;
    }
    
    void buildIndex(const Matrix<ElementType>& dataset)
    {
    	dataset_ = dataset;
    	this->buildIndex();
    }


    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        if (bestIndex_) {
            bestIndex_->addPoints(points, rebuild_threshold);
        }
    }
    
    void removePoint(size_t id)
    {
        if (bestIndex_) {
            bestIndex_->removePoint(id);
        }
    }

    
    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & target_precision_;
    	ar & build_weight_;
    	ar & memory_weight_;
    	ar & sample_fraction_;

    	flann_algorithm_t index_type;
    	if (Archive::is_saving::value) {
    		index_type = get_param<flann_algorithm_t>(bestParams_,"algorithm");
    	}
    	ar & index_type;
    	ar & bestSearchParams_.checks;

    	if (Archive::is_loading::value) {
    		bestParams_["algorithm"] = index_type;

    		index_params_["algorithm"] = getType();
            index_params_["target_precision_"] = target_precision_;
            index_params_["build_weight_"] = build_weight_;
            index_params_["memory_weight_"] = memory_weight_;
            index_params_["sample_fraction_"] = sample_fraction_;
    	}
    }

    void saveIndex(FILE* stream)
    {
    	serialization::SaveArchive sa(stream);
    	sa & *this;

    	bestIndex_->saveIndex(stream);
    }

    void loadIndex(FILE* stream)
    {
    	serialization::LoadArchive la(stream);
    	la & *this;

        IndexParams params;
        flann_algorithm_t index_type = get_param<flann_algorithm_t>(bestParams_,"algorithm");
        bestIndex_ = create_index_by_type<Distance>((flann_algorithm_t)index_type, dataset_, params, distance_);
        bestIndex_->loadIndex(stream);
    }

    int knnSearch(const Matrix<ElementType>& queries,
            Matrix<size_t>& indices,
            Matrix<DistanceType>& dists,
            size_t knn,
            const SearchParams& params) const
    {
        if (params.checks == FLANN_CHECKS_AUTOTUNED) {
            return bestIndex_->knnSearch(queries, indices, dists, knn, bestSearchParams_);
        }
        else {
            return bestIndex_->knnSearch(queries, indices, dists, knn, params);
        }
    }

    int knnSearch(const Matrix<ElementType>& queries,
            std::vector< std::vector<size_t> >& indices,
            std::vector<std::vector<DistanceType> >& dists,
            size_t knn,
            const SearchParams& params) const
    {
        if (params.checks == FLANN_CHECKS_AUTOTUNED) {
            return bestIndex_->knnSearch(queries, indices, dists, knn, bestSearchParams_);
        }
        else {
            return bestIndex_->knnSearch(queries, indices, dists, knn, params);
        }

    }
    
    int radiusSearch(const Matrix<ElementType>& queries,
            Matrix<size_t>& indices,
            Matrix<DistanceType>& dists,
            DistanceType radius,
            const SearchParams& params) const
    {
        if (params.checks == FLANN_CHECKS_AUTOTUNED) {
            return bestIndex_->radiusSearch(queries, indices, dists, radius, bestSearchParams_);
        }
        else {
            return bestIndex_->radiusSearch(queries, indices, dists, radius, params);
        }
    }

    int radiusSearch(const Matrix<ElementType>& queries,
            std::vector< std::vector<size_t> >& indices,
            std::vector<std::vector<DistanceType> >& dists,
            DistanceType radius,
            const SearchParams& params) const
    {
        if (params.checks == FLANN_CHECKS_AUTOTUNED) {
            return bestIndex_->radiusSearch(queries, indices, dists, radius, bestSearchParams_);
        }
        else {
            return bestIndex_->radiusSearch(queries, indices, dists, radius, params);
        }        
    }

    
    
    /**
     *      Method that searches for nearest-neighbors
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
        // should not get here
        assert(false);
    }

    IndexParams getParameters() const
    {
        return bestParams_;
    }

    FLANN_DEPRECATED SearchParams getSearchParameters() const
    {
        return bestSearchParams_;
    }

    FLANN_DEPRECATED float getSpeedup() const
    {
        return speedup_;
    }


    /**
     *      Number of features in this index.
     */
    size_t size() const
    {
        return bestIndex_->size();
    }

    /**
     *  The length of each vector in this index.
     */
    size_t veclen() const
    {
        return bestIndex_->veclen();
    }

    /**
     * The amount of memory (in bytes) this index uses.
     */
    int usedMemory() const
    {
        return bestIndex_->usedMemory();
    }

    /**
     * Algorithm name
     */
    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_AUTOTUNED;
    }

protected:
    void buildIndexImpl()
    {
        /* nothing to do here */
    }

    void freeIndex()
    {
        /* nothing to do here */
    }

private:

    struct CostData
    {
        float searchTimeCost;
        float buildTimeCost;
        float memoryCost;
        float totalCost;
        IndexParams params;
    };

    void evaluate_kmeans(CostData& cost)
    {
        StartStopTimer t;
        int checks;
        const int nn = 1;

        Logger::info("KMeansTree using params: max_iterations=%d, branching=%d\n",
                     get_param<int>(cost.params,"iterations"),
                     get_param<int>(cost.params,"branching"));
        KMeansIndex<Distance> kmeans(sampledDataset_, cost.params, distance_);
        // measure index build time
        t.start();
        kmeans.buildIndex();
        t.stop();
        float buildTime = (float)t.value;

        // measure search time
        float searchTime = test_index_precision(kmeans, sampledDataset_, testDataset_, gt_matches_, target_precision_, checks, distance_, nn);

        float datasetMemory = float(sampledDataset_.rows * sampledDataset_.cols * sizeof(float));
        cost.memoryCost = (kmeans.usedMemory() + datasetMemory) / datasetMemory;
        cost.searchTimeCost = searchTime;
        cost.buildTimeCost = buildTime;
        Logger::info("KMeansTree buildTime=%g, searchTime=%g, build_weight=%g\n", buildTime, searchTime, build_weight_);
    }


    void evaluate_kdtree(CostData& cost)
    {
        StartStopTimer t;
        int checks;
        const int nn = 1;

        Logger::info("KDTree using params: trees=%d\n", get_param<int>(cost.params,"trees"));
        KDTreeIndex<Distance> kdtree(sampledDataset_, cost.params, distance_);

        t.start();
        kdtree.buildIndex();
        t.stop();
        float buildTime = (float)t.value;

        //measure search time
        float searchTime = test_index_precision(kdtree, sampledDataset_, testDataset_, gt_matches_, target_precision_, checks, distance_, nn);

        float datasetMemory = float(sampledDataset_.rows * sampledDataset_.cols * sizeof(float));
        cost.memoryCost = (kdtree.usedMemory() + datasetMemory) / datasetMemory;
        cost.searchTimeCost = searchTime;
        cost.buildTimeCost = buildTime;
        Logger::info("KDTree buildTime=%g, searchTime=%g\n", buildTime, searchTime);
    }


    //    struct KMeansSimpleDownhillFunctor {
    //
    //        Autotune& autotuner;
    //        KMeansSimpleDownhillFunctor(Autotune& autotuner_) : autotuner(autotuner_) {};
    //
    //        float operator()(int* params) {
    //
    //            float maxFloat = numeric_limits<float>::max();
    //
    //            if (params[0]<2) return maxFloat;
    //            if (params[1]<0) return maxFloat;
    //
    //            CostData c;
    //            c.params["algorithm"] = KMEANS;
    //            c.params["centers-init"] = CENTERS_RANDOM;
    //            c.params["branching"] = params[0];
    //            c.params["max-iterations"] = params[1];
    //
    //            autotuner.evaluate_kmeans(c);
    //
    //            return c.timeCost;
    //
    //        }
    //    };
    //
    //    struct KDTreeSimpleDownhillFunctor {
    //
    //        Autotune& autotuner;
    //        KDTreeSimpleDownhillFunctor(Autotune& autotuner_) : autotuner(autotuner_) {};
    //
    //        float operator()(int* params) {
    //            float maxFloat = numeric_limits<float>::max();
    //
    //            if (params[0]<1) return maxFloat;
    //
    //            CostData c;
    //            c.params["algorithm"] = KDTREE;
    //            c.params["trees"] = params[0];
    //
    //            autotuner.evaluate_kdtree(c);
    //
    //            return c.timeCost;
    //
    //        }
    //    };



    void optimizeKMeans(std::vector<CostData>& costs)
    {
        Logger::info("KMEANS, Step 1: Exploring parameter space\n");

        // explore kmeans parameters space using combinations of the parameters below
        int maxIterations[] = { 1, 5, 10, 15 };
        int branchingFactors[] = { 16, 32, 64, 128, 256 };

        int kmeansParamSpaceSize = FLANN_ARRAY_LEN(maxIterations) * FLANN_ARRAY_LEN(branchingFactors);
        costs.reserve(costs.size() + kmeansParamSpaceSize);

        // evaluate kmeans for all parameter combinations
        for (size_t i = 0; i < FLANN_ARRAY_LEN(maxIterations); ++i) {
            for (size_t j = 0; j < FLANN_ARRAY_LEN(branchingFactors); ++j) {
                CostData cost;
                cost.params["algorithm"] = FLANN_INDEX_KMEANS;
                cost.params["centers_init"] = FLANN_CENTERS_RANDOM;
                cost.params["iterations"] = maxIterations[i];
                cost.params["branching"] = branchingFactors[j];

                evaluate_kmeans(cost);
                costs.push_back(cost);
            }
        }

        //         Logger::info("KMEANS, Step 2: simplex-downhill optimization\n");
        //
        //         const int n = 2;
        //         // choose initial simplex points as the best parameters so far
        //         int kmeansNMPoints[n*(n+1)];
        //         float kmeansVals[n+1];
        //         for (int i=0;i<n+1;++i) {
        //             kmeansNMPoints[i*n] = (int)kmeansCosts[i].params["branching"];
        //             kmeansNMPoints[i*n+1] = (int)kmeansCosts[i].params["max-iterations"];
        //             kmeansVals[i] = kmeansCosts[i].timeCost;
        //         }
        //         KMeansSimpleDownhillFunctor kmeans_cost_func(*this);
        //         // run optimization
        //         optimizeSimplexDownhill(kmeansNMPoints,n,kmeans_cost_func,kmeansVals);
        //         // store results
        //         for (int i=0;i<n+1;++i) {
        //             kmeansCosts[i].params["branching"] = kmeansNMPoints[i*2];
        //             kmeansCosts[i].params["max-iterations"] = kmeansNMPoints[i*2+1];
        //             kmeansCosts[i].timeCost = kmeansVals[i];
        //         }
    }


    void optimizeKDTree(std::vector<CostData>& costs)
    {
        Logger::info("KD-TREE, Step 1: Exploring parameter space\n");

        // explore kd-tree parameters space using the parameters below
        int testTrees[] = { 1, 4, 8, 16, 32 };

        // evaluate kdtree for all parameter combinations
        for (size_t i = 0; i < FLANN_ARRAY_LEN(testTrees); ++i) {
            CostData cost;
            cost.params["algorithm"] = FLANN_INDEX_KDTREE;
            cost.params["trees"] = testTrees[i];

            evaluate_kdtree(cost);
            costs.push_back(cost);
        }

        //         Logger::info("KD-TREE, Step 2: simplex-downhill optimization\n");
        //
        //         const int n = 1;
        //         // choose initial simplex points as the best parameters so far
        //         int kdtreeNMPoints[n*(n+1)];
        //         float kdtreeVals[n+1];
        //         for (int i=0;i<n+1;++i) {
        //             kdtreeNMPoints[i] = (int)kdtreeCosts[i].params["trees"];
        //             kdtreeVals[i] = kdtreeCosts[i].timeCost;
        //         }
        //         KDTreeSimpleDownhillFunctor kdtree_cost_func(*this);
        //         // run optimization
        //         optimizeSimplexDownhill(kdtreeNMPoints,n,kdtree_cost_func,kdtreeVals);
        //         // store results
        //         for (int i=0;i<n+1;++i) {
        //             kdtreeCosts[i].params["trees"] = kdtreeNMPoints[i];
        //             kdtreeCosts[i].timeCost = kdtreeVals[i];
        //         }
    }

    /**
     *  Chooses the best nearest-neighbor algorithm and estimates the optimal
     *  parameters to use when building the index (for a given precision).
     *  Returns a dictionary with the optimal parameters.
     */
    IndexParams estimateBuildParams()
    {
        std::vector<CostData> costs;

        int sampleSize = int(sample_fraction_ * dataset_.rows);
        int testSampleSize = std::min(sampleSize / 10, 1000);

        Logger::info("Entering autotuning, dataset size: %d, sampleSize: %d, testSampleSize: %d, target precision: %g\n", dataset_.rows, sampleSize, testSampleSize, target_precision_);

        // For a very small dataset, it makes no sense to build any fancy index, just
        // use linear search
        if (testSampleSize < 10) {
            Logger::info("Choosing linear, dataset too small\n");
            return LinearIndexParams();
        }

        // We use a fraction of the original dataset to speedup the autotune algorithm
        sampledDataset_ = random_sample(dataset_, sampleSize);
        // We use a cross-validation approach, first we sample a testset from the dataset
        testDataset_ = random_sample(sampledDataset_, testSampleSize, true);

        // We compute the ground truth using linear search
        Logger::info("Computing ground truth... \n");
        gt_matches_ = Matrix<size_t>(new size_t[testDataset_.rows], testDataset_.rows, 1);
        StartStopTimer t;
        int repeats = 0;
        t.reset();
        while (t.value<0.2) {
        	repeats++;
            t.start();
        	compute_ground_truth<Distance>(sampledDataset_, testDataset_, gt_matches_, 0, distance_);
            t.stop();
        }

        CostData linear_cost;
        linear_cost.searchTimeCost = (float)t.value/repeats;
        linear_cost.buildTimeCost = 0;
        linear_cost.memoryCost = 0;
        linear_cost.params["algorithm"] = FLANN_INDEX_LINEAR;

        costs.push_back(linear_cost);

        // Start parameter autotune process
        Logger::info("Autotuning parameters...\n");

        optimizeKMeans(costs);
        optimizeKDTree(costs);

        float bestTimeCost = costs[0].buildTimeCost * build_weight_ + costs[0].searchTimeCost;
        for (size_t i = 0; i < costs.size(); ++i) {
            float timeCost = costs[i].buildTimeCost * build_weight_ + costs[i].searchTimeCost;
            Logger::debug("Time cost: %g\n", timeCost);
            if (timeCost < bestTimeCost) {
                bestTimeCost = timeCost;
            }
        }
        Logger::debug("Best time cost: %g\n", bestTimeCost);

    	IndexParams bestParams = costs[0].params;
        if (bestTimeCost > 0) {
        	float bestCost = (costs[0].buildTimeCost * build_weight_ + costs[0].searchTimeCost) / bestTimeCost;
        	for (size_t i = 0; i < costs.size(); ++i) {
        		float crtCost = (costs[i].buildTimeCost * build_weight_ + costs[i].searchTimeCost) / bestTimeCost +
        				memory_weight_ * costs[i].memoryCost;
        		Logger::debug("Cost: %g\n", crtCost);
        		if (crtCost < bestCost) {
        			bestCost = crtCost;
        			bestParams = costs[i].params;
        		}
        	}
            Logger::debug("Best cost: %g\n", bestCost);
        }

        delete[] gt_matches_.ptr();
        delete[] testDataset_.ptr();
        delete[] sampledDataset_.ptr();

        return bestParams;
    }



    /**
     *  Estimates the search time parameters needed to get the desired precision.
     *  Precondition: the index is built
     *  Postcondition: the searchParams will have the optimum params set, also the speedup obtained over linear search.
     */
    float estimateSearchParams(SearchParams& searchParams)
    {
        const int nn = 1;
        const size_t SAMPLE_COUNT = 1000;

        assert(bestIndex_ != NULL); // must have a valid index

        float speedup = 0;

        int samples = (int)std::min(dataset_.rows / 10, SAMPLE_COUNT);
        if (samples > 0) {
            Matrix<ElementType> testDataset = random_sample(dataset_, samples);

            Logger::info("Computing ground truth\n");

            // we need to compute the ground truth first
            Matrix<size_t> gt_matches(new size_t[testDataset.rows], testDataset.rows, 1);
            StartStopTimer t;
            int repeats = 0;
            t.reset();
            while (t.value<0.2) {
            	repeats++;
                t.start();
            	compute_ground_truth<Distance>(dataset_, testDataset, gt_matches, 1, distance_);
                t.stop();
            }
            float linear = (float)t.value/repeats;

            int checks;
            Logger::info("Estimating number of checks\n");

            float searchTime;
            float cb_index;
            if (bestIndex_->getType() == FLANN_INDEX_KMEANS) {
                Logger::info("KMeans algorithm, estimating cluster border factor\n");
                KMeansIndex<Distance>* kmeans = static_cast<KMeansIndex<Distance>*>(bestIndex_);
                float bestSearchTime = -1;
                float best_cb_index = -1;
                int best_checks = -1;
                for (cb_index = 0; cb_index < 1.1f; cb_index += 0.2f) {
                    kmeans->set_cb_index(cb_index);
                    searchTime = test_index_precision(*kmeans, dataset_, testDataset, gt_matches, target_precision_, checks, distance_, nn, 1);
                    if ((searchTime < bestSearchTime) || (bestSearchTime == -1)) {
                        bestSearchTime = searchTime;
                        best_cb_index = cb_index;
                        best_checks = checks;
                    }
                }
                searchTime = bestSearchTime;
                cb_index = best_cb_index;
                checks = best_checks;

                kmeans->set_cb_index(best_cb_index);
                Logger::info("Optimum cb_index: %g\n", cb_index);
                bestParams_["cb_index"] = cb_index;
            }
            else {
                searchTime = test_index_precision(*bestIndex_, dataset_, testDataset, gt_matches, target_precision_, checks, distance_, nn, 1);
            }

            Logger::info("Required number of checks: %d \n", checks);
            searchParams.checks = checks;

            speedup = linear / searchTime;

            delete[] gt_matches.ptr();
            delete[] testDataset.ptr();
        }

        return speedup;
    }


    void swap(AutotunedIndex& other)
    {
    	BaseClass::swap(other);
    	std::swap(bestIndex_, other.bestIndex_);
    	std::swap(bestParams_, other.bestParams_);
    	std::swap(bestSearchParams_, other.bestSearchParams_);
    	std::swap(speedup_, other.speedup_);
    	std::swap(dataset_, other.dataset_);
    	std::swap(target_precision_, other.target_precision_);
    	std::swap(build_weight_, other.build_weight_);
    	std::swap(memory_weight_, other.memory_weight_);
    	std::swap(sample_fraction_, other.sample_fraction_);
    }

private:
    NNIndex<Distance>* bestIndex_;

    IndexParams bestParams_;
    SearchParams bestSearchParams_;

    Matrix<ElementType> sampledDataset_;
    Matrix<ElementType> testDataset_;
    Matrix<size_t> gt_matches_;

    float speedup_;

    /**
     * The dataset used by this index
     */
    Matrix<ElementType> dataset_;

    /**
     * Index parameters
     */
    float target_precision_;
    float build_weight_;
    float memory_weight_;
    float sample_fraction_;

    USING_BASECLASS_SYMBOLS
};
}

#endif /* FLANN_AUTOTUNED_INDEX_H_ */
