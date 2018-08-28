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

#ifndef FLANN_HEAP_H_
#define FLANN_HEAP_H_

#include <algorithm>
#include <vector>

namespace flann
{

/**
 * Priority Queue Implementation
 *
 * The priority queue is implemented with a heap.  A heap is a complete
 * (full) binary tree in which each parent is less than both of its
 * children, but the order of the children is unspecified.
 */
template <typename T>
class Heap
{

    /**
     * Storage array for the heap.
     * Type T must be comparable.
     */
    std::vector<T> heap;
    int length;

    /**
     * Number of element in the heap
     */
    int count;



public:
    /**
     * Constructor.
     *
     * Params:
     *     size = heap size
     */

    Heap(int size)
    {
        length = size;
        heap.reserve(length);
        count = 0;
    }

    /**
     *
     * Returns: heap size
     */
    int size()
    {
        return count;
    }

    /**
     * Tests if the heap is empty
     *
     * Returns: true is heap empty, false otherwise
     */
    bool empty()
    {
        return size()==0;
    }

    /**
     * Clears the heap.
     */
    void clear()
    {
        heap.clear();
        count = 0;
    }

    struct CompareT : public std::binary_function<T,T,bool>
    {
        bool operator()(const T& t_1, const T& t_2) const
        {
            return t_2 < t_1;
        }
    };

    /**
     * Insert a new element in the heap.
     *
     * We select the next empty leaf node, and then keep moving any larger
     * parents down until the right location is found to store this element.
     *
     * Params:
     *     value = the new element to be inserted in the heap
     */
    void insert(const T& value)
    {
        /* If heap is full, then return without adding this element. */
        if (count == length) {
            return;
        }

        heap.push_back(value);
        static CompareT compareT;
        std::push_heap(heap.begin(), heap.end(), compareT);
        ++count;
    }



    /**
     * Returns the node of minimum value from the heap (top of the heap).
     *
     * Params:
     *     value = out parameter used to return the min element
     * Returns: false if heap empty
     */
    bool popMin(T& value)
    {
        if (count == 0) {
            return false;
        }

        value = heap[0];
        static CompareT compareT;
        std::pop_heap(heap.begin(), heap.end(), compareT);
        heap.pop_back();
        --count;

        return true;  /* Return old last node. */
    }
};


template <typename T>
class IntervalHeap
{
	struct Interval
	{
		T left;
		T right;
	};

    /**
     * Storage array for the heap.
     * Type T must be comparable.
     */
    std::vector<Interval> heap;
    size_t capacity_;
    size_t size_;

public:
    /**
     * Constructor.
     *
     * Params:
     *     size = heap size
     */

    IntervalHeap(int capacity) : capacity_(capacity), size_(0)
    {
        heap.resize(capacity/2 + capacity%2 + 1); // 1-based indexing
    }

    /**
     * @return Heap size
     */
    size_t size()
    {
        return size_;
    }

    /**
     * Tests if the heap is empty
     * @return true is heap empty, false otherwise
     */
    bool empty()
    {
        return size_==0;
    }

    /**
     * Clears the heap.
     */
    void clear()
    {
        size_ = 0;
    }

    void insert(const T& value)
    {
        /* If heap is full, then return without adding this element. */
        if (size_ == capacity_) {
            return;
        }

        // insert into the root
        if (size_<2) {
        	if (size_==0) {
        		heap[1].left = value;
        		heap[1].right = value;
        	}
        	else {
        		if (value<heap[1].left) {
        			heap[1].left = value;
        		}
        		else {
        			heap[1].right = value;
        		}
        	}
        	++size_;
        	return;
        }

        size_t last_pos = size_/2 + size_%2;
        bool min_heap;

        if (size_%2) { // odd number of elements
        	min_heap = (value<heap[last_pos].left)? true : false;
        }
        else {
        	++last_pos;
        	min_heap = (value<heap[last_pos/2].left)? true : false;
        }

        if (min_heap) {
        	size_t pos = last_pos;
        	size_t par = pos/2;
        	while (pos>1 && value < heap[par].left) {
        		heap[pos].left = heap[par].left;
        		pos = par;
        		par = pos/2;
        	}
        	heap[pos].left = value;
        	++size_;

        	if (size_%2) { // duplicate element in last position if size is odd
        		heap[last_pos].right = heap[last_pos].left;
        	}
        }
        else {
        	size_t pos = last_pos;
        	size_t par = pos/2;
        	while (pos>1 && heap[par].right < value) {
        		heap[pos].right = heap[par].right;
        		pos = par;
        		par = pos/2;
        	}
        	heap[pos].right = value;
        	++size_;

        	if (size_%2) { // duplicate element in last position if size is odd
        		heap[last_pos].left = heap[last_pos].right;
        	}
        }
    }


    /**
     * Returns the node of minimum value from the heap
     * @param value out parameter used to return the min element
     * @return false if heap empty
     */
    bool popMin(T& value)
    {
        if (size_ == 0) {
            return false;
        }

        value = heap[1].left;
        size_t last_pos = size_/2 + size_%2;
        T elem = heap[last_pos].left;

        if (size_ % 2) { // odd number of elements
        	--last_pos;
        }
        else {
        	heap[last_pos].left = heap[last_pos].right;
        }
        --size_;
        if (size_<2) return true;

        size_t crt=1; // root node
        size_t child = crt*2;

        while (child <= last_pos) {
        	if (child < last_pos && heap[child+1].left < heap[child].left) ++child; // pick the child with min

        	if (!(heap[child].left<elem)) break;

        	heap[crt].left = heap[child].left;
        	if (heap[child].right<elem) {
        		std::swap(elem, heap[child].right);
        	}

        	crt = child;
    		child *= 2;
        }
        heap[crt].left = elem;
        return true;
    }


    /**
     * Returns the element of maximum value from the heap
     * @param value
     * @return false if heap empty
     */
    bool popMax(T& value)
    {
        if (size_ == 0) {
            return false;
        }

        value = heap[1].right;
        size_t last_pos = size_/2 + size_%2;
        T elem = heap[last_pos].right;

        if (size_%2) { // odd number of elements
        	--last_pos;
        }
        else {
        	heap[last_pos].right = heap[last_pos].left;
        }
        --size_;
        if (size_<2) return true;

        size_t crt=1; // root node
        size_t child = crt*2;

        while (child <= last_pos) {
        	if (child < last_pos && heap[child].right < heap[child+1].right) ++child; // pick the child with max

        	if (!(elem < heap[child].right)) break;

        	heap[crt].right = heap[child].right;
        	if (elem<heap[child].left) {
        		std::swap(elem, heap[child].left);
        	}

        	crt = child;
    		child *= 2;
        }
        heap[crt].right = elem;
        return true;
    }


    bool getMin(T& value)
    {
    	if (size_==0) {
    		return false;
    	}
    	value = heap[1].left;
    	return true;
    }


    bool getMax(T& value)
    {
    	if (size_==0) {
    		return false;
    	}
    	value = heap[1].right;
    	return true;
    }
};


template <typename T>
class BoundedHeap
{
	IntervalHeap<T> interval_heap_;
	size_t capacity_;
public:
	BoundedHeap(size_t capacity) : interval_heap_(capacity), capacity_(capacity)
	{

	}

    /**
     * Returns: heap size
     */
    int size()
    {
        return interval_heap_.size();
    }

    /**
     * Tests if the heap is empty
     * Returns: true is heap empty, false otherwise
     */
    bool empty()
    {
        return interval_heap_.empty();
    }

    /**
     * Clears the heap.
     */
    void clear()
    {
    	interval_heap_.clear();
    }

    void insert(const T& value)
    {
    	if (interval_heap_.size()==capacity_) {
    		T max;
    		interval_heap_.getMax(max);
    		if (max<value) return;
   			interval_heap_.popMax(max);
    	}
    	interval_heap_.insert(value);
    }

    bool popMin(T& value)
    {
    	return interval_heap_.popMin(value);
    }
};



}

#endif //FLANN_HEAP_H_
