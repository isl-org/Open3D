/*
 * center_chooser.h
 *
 *  Created on: 2012-11-04
 *      Author: marius
 */

#ifndef CENTER_CHOOSER_H_
#define CENTER_CHOOSER_H_

#include <flann/util/matrix.h>

namespace flann
{

template <typename Distance>
class CenterChooser
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    CenterChooser(const Distance& distance) : distance_(distance) {};

    void setDataset(const flann::Matrix<ElementType>& dataset)
    {
    	dataset_ = dataset;
    }

    virtual ~CenterChooser() {};

    /**
     * Chooses cluster centers
     *
     * @param k number of centers to choose
     * @param indices indices of points to choose the centers from
     * @param indices_length length of indices
     * @param centers indices of chosen centers
     * @param centers_length length of centers array
     */
	virtual void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length) = 0;

protected:
	flann::Matrix<ElementType> dataset_;
	Distance distance_;
};


template <typename Distance>
class RandomCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    using CenterChooser<Distance>::dataset_;
    using CenterChooser<Distance>::distance_;

    RandomCenterChooser(const Distance& distance) :
    	CenterChooser<Distance>(distance) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        UniqueRandom r(indices_length);

        int index;
        for (index=0; index<k; ++index) {
            bool duplicate = true;
            int rnd;
            while (duplicate) {
                duplicate = false;
                rnd = r.next();
                if (rnd<0) {
                    centers_length = index;
                    return;
                }

                centers[index] = indices[rnd];

                for (int j=0; j<index; ++j) {
                    DistanceType sq = distance_(dataset_[centers[index]], dataset_[centers[j]], dataset_.cols);
                    if (sq<1e-16) {
                        duplicate = true;
                    }
                }
            }
        }

        centers_length = index;
    }
};



/**
 * Chooses the initial centers using the Gonzales algorithm.
 */
template <typename Distance>
class GonzalesCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    using CenterChooser<Distance>::dataset_;
    using CenterChooser<Distance>::distance_;

    GonzalesCenterChooser(const Distance& distance) : CenterChooser<Distance>( distance) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        int rnd = rand_int(n);
        assert(rnd >=0 && rnd < n);

        centers[0] = indices[rnd];

        int index;
        for (index=1; index<k; ++index) {

            int best_index = -1;
            DistanceType best_val = 0;
            for (int j=0; j<n; ++j) {
            	DistanceType dist = distance_(dataset_[centers[0]],dataset_[indices[j]],dataset_.cols);
                for (int i=1; i<index; ++i) {
                    DistanceType tmp_dist = distance_(dataset_[centers[i]],dataset_[indices[j]],dataset_.cols);
                    if (tmp_dist<dist) {
                        dist = tmp_dist;
                    }
                }
                if (dist>best_val) {
                    best_val = dist;
                    best_index = j;
                }
            }
            if (best_index!=-1) {
                centers[index] = indices[best_index];
            }
            else {
                break;
            }
        }
        centers_length = index;
    }
};


/**
 * Chooses the initial centers using the algorithm proposed in the KMeans++ paper:
 * Arthur, David; Vassilvitskii, Sergei - k-means++: The Advantages of Careful Seeding
 */
template <typename Distance>
class KMeansppCenterChooser : public CenterChooser<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    using CenterChooser<Distance>::dataset_;
    using CenterChooser<Distance>::distance_;

    KMeansppCenterChooser(const Distance& distance) : CenterChooser<Distance>(distance) {}

    void operator()(int k, int* indices, int indices_length, int* centers, int& centers_length)
    {
        int n = indices_length;

        double currentPot = 0;
        DistanceType* closestDistSq = new DistanceType[n];

        // Choose one random center and set the closestDistSq values
        int index = rand_int(n);
        assert(index >=0 && index < n);
        centers[0] = indices[index];

        for (int i = 0; i < n; i++) {
            closestDistSq[i] = distance_(dataset_[indices[i]], dataset_[indices[index]], dataset_.cols);
            currentPot += closestDistSq[i];
        }


        const int numLocalTries = 1;

        // Choose each center
        int centerCount;
        for (centerCount = 1; centerCount < k; centerCount++) {

            // Repeat several trials
            double bestNewPot = -1;
            int bestNewIndex = 0;
            for (int localTrial = 0; localTrial < numLocalTries; localTrial++) {

                // Choose our center - have to be slightly careful to return a valid answer even accounting
                // for possible rounding errors
                double randVal = rand_double(currentPot);
                for (index = 0; index < n-1; index++) {
                    if (randVal <= closestDistSq[index]) break;
                    else randVal -= closestDistSq[index];
                }

                // Compute the new potential
                double newPot = 0;
                for (int i = 0; i < n; i++) newPot += std::min( distance_(dataset_[indices[i]], dataset_[indices[index]], dataset_.cols), closestDistSq[i] );

                // Store the best result
                if ((bestNewPot < 0)||(newPot < bestNewPot)) {
                    bestNewPot = newPot;
                    bestNewIndex = index;
                }
            }

            // Add the appropriate center
            centers[centerCount] = indices[bestNewIndex];
            currentPot = bestNewPot;
            for (int i = 0; i < n; i++) closestDistSq[i] = std::min( distance_(dataset_[indices[i]], dataset_[indices[bestNewIndex]], dataset_.cols), closestDistSq[i] );
        }

        centers_length = centerCount;

        delete[] closestDistSq;
    }
};


}


#endif /* CENTER_CHOOSER_H_ */
