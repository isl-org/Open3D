/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Matrix data structures and parsing logic
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cstring>

#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <fstream>
#include <stdio.h>

#ifdef CUB_MKL
    #include <numa.h>
    #include <mkl.h>
#endif

using namespace std;

/******************************************************************************
 * COO matrix type
 ******************************************************************************/

struct GraphStats
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;

    double      diag_dist_mean;         // mean
    double      diag_dist_std_dev;      // sample std dev
    double      pearson_r;    // coefficient of variation

    double      row_length_mean;        // mean
    double      row_length_std_dev;     // sample std_dev
    double      row_length_variation;   // coefficient of variation
    double      row_length_skewness;    // skewness

    void Display(bool show_labels = true)
    {
        if (show_labels)
            printf("\n"
                "\t num_rows: %d\n"
                "\t num_cols: %d\n"
                "\t num_nonzeros: %d\n"
                "\t diag_dist_mean: %.2f\n"
                "\t diag_dist_std_dev: %.2f\n"
                "\t pearson_r: %f\n"
                "\t row_length_mean: %.5f\n"
                "\t row_length_std_dev: %.5f\n"
                "\t row_length_variation: %.5f\n"
                "\t row_length_skewness: %.5f\n",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    diag_dist_mean,
                    diag_dist_std_dev,
                    pearson_r,
                    row_length_mean,
                    row_length_std_dev,
                    row_length_variation,
                    row_length_skewness);
        else
            printf(
                "%d, "
                "%d, "
                "%d, "
                "%.2f, "
                "%.2f, "
                "%f, "
                "%.5f, "
                "%.5f, "
                "%.5f, "
                "%.5f, ",
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    diag_dist_mean,
                    diag_dist_std_dev,
                    pearson_r,
                    row_length_mean,
                    row_length_std_dev,
                    row_length_variation,
                    row_length_skewness);
    }
};



/******************************************************************************
 * COO matrix type
 ******************************************************************************/


/**
 * COO matrix type.  A COO matrix is just a vector of edge tuples.  Tuples are sorted
 * first by row, then by column.
 */
template<typename ValueT, typename OffsetT>
struct CooMatrix
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // COO edge tuple
    struct CooTuple
    {
        OffsetT            row;
        OffsetT            col;
        ValueT             val;

        CooTuple() {}
        CooTuple(OffsetT row, OffsetT col) : row(row), col(col) {}
        CooTuple(OffsetT row, OffsetT col, ValueT val) : row(row), col(col), val(val) {}

        /**
         * Comparator for sorting COO sparse format num_nonzeros
         */
        bool operator<(const CooTuple &other) const
        {
            if ((row < other.row) || ((row == other.row) && (col < other.col)))
            {
                return true;
            }

            return false;
        }
    };


    //---------------------------------------------------------------------
    // Data members
    //---------------------------------------------------------------------

    // Fields
    int                 num_rows;
    int                 num_cols;
    int                 num_nonzeros;
    CooTuple*           coo_tuples;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    // Constructor
    CooMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), coo_tuples(NULL) {}


    /**
     * Clear
     */
    void Clear()
    {
        if (coo_tuples) delete[] coo_tuples;
        coo_tuples = NULL;
    }


    // Destructor
    ~CooMatrix()
    {
        Clear();
    }


    // Display matrix to stdout
    void Display()
    {
        cout << "COO Matrix (" << num_rows << " rows, " << num_cols << " columns, " << num_nonzeros << " non-zeros):\n";
        cout << "Ordinal, Row, Column, Value\n";
        for (int i = 0; i < num_nonzeros; i++)
        {
            cout << '\t' << i << ',' << coo_tuples[i].row << ',' << coo_tuples[i].col << ',' << coo_tuples[i].val << "\n";
        }
    }


    /**
     * Builds a symmetric COO sparse from an asymmetric CSR matrix.
     */
    template <typename CsrMatrixT>
    void InitCsrSymmetric(CsrMatrixT &csr_matrix)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = csr_matrix.num_cols;
        num_cols        = csr_matrix.num_rows;
        num_nonzeros    = csr_matrix.num_nonzeros * 2;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < csr_matrix.num_rows; ++row)
        {
            for (OffsetT nonzero = csr_matrix.row_offsets[row]; nonzero < csr_matrix.row_offsets[row + 1]; ++nonzero)
            {
                coo_tuples[nonzero].row = row;
                coo_tuples[nonzero].col = csr_matrix.column_indices[nonzero];
                coo_tuples[nonzero].val = csr_matrix.values[nonzero];

                coo_tuples[csr_matrix.num_nonzeros + nonzero].row = coo_tuples[nonzero].col;
                coo_tuples[csr_matrix.num_nonzeros + nonzero].col = coo_tuples[nonzero].row;
                coo_tuples[csr_matrix.num_nonzeros + nonzero].val = csr_matrix.values[nonzero];

            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);
    }

    /**
     * Builds a COO sparse from a relabeled CSR matrix.
     */
    template <typename CsrMatrixT>
    void InitCsrRelabel(CsrMatrixT &csr_matrix, OffsetT* relabel_indices)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = csr_matrix.num_rows;
        num_cols        = csr_matrix.num_cols;
        num_nonzeros    = csr_matrix.num_nonzeros;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT nonzero = csr_matrix.row_offsets[row]; nonzero < csr_matrix.row_offsets[row + 1]; ++nonzero)
            {
                coo_tuples[nonzero].row = relabel_indices[row];
                coo_tuples[nonzero].col = relabel_indices[csr_matrix.column_indices[nonzero]];
                coo_tuples[nonzero].val = csr_matrix.values[nonzero];
            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);
    }



    /**
     * Builds a METIS COO sparse from the given file.
     */
    void InitMetis(const string &metis_filename)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        // TODO
    }


    /**
     * Builds a MARKET COO sparse from the given file.
     */
    void InitMarket(
        const string&   market_filename,
        ValueT          default_value       = 1.0,
        bool            verbose             = false)
    {
        if (verbose) {
            printf("Reading... "); fflush(stdout);
        }

        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        std::ifstream ifs;
        ifs.open(market_filename.c_str(), std::ifstream::in);
        if (!ifs.good())
        {
            fprintf(stderr, "Error opening file\n");
            exit(1);
        }

        bool    array = false;
        bool    symmetric = false;
        bool    skew = false;
        int     current_edge = -1;
        char    line[1024];

        if (verbose) {
            printf("Parsing... "); fflush(stdout);
        }

        while (true)
        {
            ifs.getline(line, 1024);
            if (!ifs.good())
            {
                // Done
                break;
            }

            if (line[0] == '%')
            {
                // Comment
                if (line[1] == '%')
                {
                    // Banner
                    symmetric   = (strstr(line, "symmetric") != NULL);
                    skew        = (strstr(line, "skew") != NULL);
                    array       = (strstr(line, "array") != NULL);

                    if (verbose) {
                        printf("(symmetric: %d, skew: %d, array: %d) ", symmetric, skew, array); fflush(stdout);
                    }
                }
            }
            else if (current_edge == -1)
            {
                // Problem description
                int nparsed = sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_nonzeros);
                if ((!array) && (nparsed == 3))
                {
                    if (symmetric)
                        num_nonzeros *= 2;

                    // Allocate coo matrix
                    coo_tuples = new CooTuple[num_nonzeros];
                    current_edge = 0;

                }
                else if (array && (nparsed == 2))
                {
                    // Allocate coo matrix
                    num_nonzeros = num_rows * num_cols;
                    coo_tuples = new CooTuple[num_nonzeros];
                    current_edge = 0;
                }
                else
                {
                    fprintf(stderr, "Error parsing MARKET matrix: invalid problem description: %s\n", line);
                    exit(1);
                }

            }
            else
            {
                // Edge
                if (current_edge >= num_nonzeros)
                {
                    fprintf(stderr, "Error parsing MARKET matrix: encountered more than %d num_nonzeros\n", num_nonzeros);
                    exit(1);
                }

                int row, col;
                double val;

                if (array)
                {
                    if (sscanf(line, "%lf", &val) != 1)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed current_edge: '%s' at edge %d\n", line, current_edge);
                        exit(1);
                    }
                    col = (current_edge / num_rows);
                    row = (current_edge - (num_rows * col));

                    coo_tuples[current_edge] = CooTuple(row, col, val);    // Convert indices to zero-based
                }
                else
                {
                    // Parse nonzero (note: using strtol and strtod is 2x faster than sscanf or istream parsing)
                    char *l = line;
                    char *t = NULL;

                    // parse row
                    row = strtol(l, &t, 0);
                    if (t == l)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed row at edge %d\n", current_edge);
                        exit(1);
                    }
                    l = t;

                    // parse col
                    col = strtol(l, &t, 0);
                    if (t == l)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix: badly formed col at edge %d\n", current_edge);
                        exit(1);
                    }
                    l = t;

                    // parse val
                    val = strtod(l, &t);
                    if (t == l)
                    {
                        val = default_value;
                    }
/*
                    int nparsed = sscanf(line, "%d %d %lf", &row, &col, &val);
                    if (nparsed == 2)
                    {
                        // No value specified
                        val = default_value;
                        
                    }
                    else if (nparsed != 3)
                    {
                        fprintf(stderr, "Error parsing MARKET matrix 1: badly formed current_edge: %d parsed at edge %d\n", nparsed, current_edge);
                        exit(1);
                    }
*/

                    coo_tuples[current_edge] = CooTuple(row - 1, col - 1, val);    // Convert indices to zero-based

                }

                current_edge++;

                if (symmetric && (row != col))
                {
                    coo_tuples[current_edge].row = coo_tuples[current_edge - 1].col;
                    coo_tuples[current_edge].col = coo_tuples[current_edge - 1].row;
                    coo_tuples[current_edge].val = coo_tuples[current_edge - 1].val * (skew ? -1 : 1);
                    current_edge++;
                }
            }
        }

        // Adjust nonzero count (nonzeros along the diagonal aren't reversed)
        num_nonzeros = current_edge;

        if (verbose) {
            printf("done. Ordering..."); fflush(stdout);
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        if (verbose) {
            printf("done. "); fflush(stdout);
        }

        ifs.close();
    }


    /**
     * Builds a dense matrix
     */
    int InitDense(
        OffsetT     num_rows,
        OffsetT     num_cols,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        this->num_rows  = num_rows;
        this->num_cols  = num_cols;

        num_nonzeros    = num_rows * num_cols;
        coo_tuples      = new CooTuple[num_nonzeros];

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            for (OffsetT col = 0; col < num_cols; ++col)
            {
                coo_tuples[(row * num_cols) + col] = CooTuple(row, col, default_value);
            }
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }

    /**
     * Builds a wheel COO sparse matrix having spokes spokes.
     */
    int InitWheel(
        OffsetT     spokes,
        ValueT      default_value   = 1.0,
        bool        verbose         = false)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        num_rows        = spokes + 1;
        num_cols        = num_rows;
        num_nonzeros    = spokes * 2;
        coo_tuples      = new CooTuple[num_nonzeros];

        // Add spoke num_nonzeros
        int current_edge = 0;
        for (OffsetT i = 0; i < spokes; i++)
        {
            coo_tuples[current_edge] = CooTuple(0, i + 1, default_value);
            current_edge++;
        }

        // Add rim
        for (OffsetT i = 0; i < spokes; i++)
        {
            OffsetT dest = (i + 1) % spokes;
            coo_tuples[current_edge] = CooTuple(i + 1, dest + 1, default_value);
            current_edge++;
        }

        // Sort by rows, then columns
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }


    /**
     * Builds a square 2D grid CSR matrix.  Interior num_vertices have degree 5 when including
     * a self-loop.
     *
     * Returns 0 on success, 1 on failure.
     */
    int InitGrid2d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            exit(1);
        }

        int     interior_nodes  = (width - 2) * (width - 2);
        int     edge_nodes      = (width - 2) * 4;
        int     corner_nodes    = 4;
        num_rows                       = width * width;
        num_cols                       = num_rows;
        num_nonzeros                   = (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples          = new CooTuple[num_nonzeros];
        int current_edge    = 0;

        for (OffsetT j = 0; j < width; j++)
        {
            for (OffsetT k = 0; k < width; k++)
            {
                OffsetT me = (j * width) + k;

                // West
                OffsetT neighbor = (j * width) + (k - 1);
                if (k - 1 >= 0) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // East
                neighbor = (j * width) + (k + 1);
                if (k + 1 < width) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // North
                neighbor = ((j - 1) * width) + k;
                if (j - 1 >= 0) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                // South
                neighbor = ((j + 1) * width) + k;
                if (j + 1 < width) {
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }

                if (self_loop)
                {
                    neighbor = me;
                    coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                    current_edge++;
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }


    /**
     * Builds a square 3D grid COO sparse matrix.  Interior num_vertices have degree 7 when including
     * a self-loop.  Values are unintialized, coo_tuples are sorted.
     */
    int InitGrid3d(OffsetT width, bool self_loop, ValueT default_value = 1.0)
    {
        if (coo_tuples)
        {
            fprintf(stderr, "Matrix already constructed\n");
            return -1;
        }

        OffsetT interior_nodes  = (width - 2) * (width - 2) * (width - 2);
        OffsetT face_nodes      = (width - 2) * (width - 2) * 6;
        OffsetT edge_nodes      = (width - 2) * 12;
        OffsetT corner_nodes    = 8;
        num_cols                       = width * width * width;
        num_rows                       = num_cols;
        num_nonzeros                     = (interior_nodes * 6) + (face_nodes * 5) + (edge_nodes * 4) + (corner_nodes * 3);

        if (self_loop)
            num_nonzeros += num_rows;

        coo_tuples          = new CooTuple[num_nonzeros];
        int current_edge    = 0;

        for (OffsetT i = 0; i < width; i++)
        {
            for (OffsetT j = 0; j < width; j++)
            {
                for (OffsetT k = 0; k < width; k++)
                {

                    OffsetT me = (i * width * width) + (j * width) + k;

                    // Up
                    OffsetT neighbor = (i * width * width) + (j * width) + (k - 1);
                    if (k - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // Down
                    neighbor = (i * width * width) + (j * width) + (k + 1);
                    if (k + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // West
                    neighbor = (i * width * width) + ((j - 1) * width) + k;
                    if (j - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // East
                    neighbor = (i * width * width) + ((j + 1) * width) + k;
                    if (j + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // North
                    neighbor = ((i - 1) * width * width) + (j * width) + k;
                    if (i - 1 >= 0) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    // South
                    neighbor = ((i + 1) * width * width) + (j * width) + k;
                    if (i + 1 < width) {
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }

                    if (self_loop)
                    {
                        neighbor = me;
                        coo_tuples[current_edge] = CooTuple(me, neighbor, default_value);
                        current_edge++;
                    }
                }
            }
        }

        // Sort by rows, then columns, update dims
        std::stable_sort(coo_tuples, coo_tuples + num_nonzeros);

        return 0;
    }
};



/******************************************************************************
 * COO matrix type
 ******************************************************************************/


/**
 * CSR sparse format matrix
 */
template<
    typename ValueT,
    typename OffsetT>
struct CsrMatrix
{
    int         num_rows;
    int         num_cols;
    int         num_nonzeros;
    OffsetT*    row_offsets;
    OffsetT*    column_indices;
    ValueT*     values;
    bool        numa_malloc;

    /**
     * Constructor
     */
    CsrMatrix() : num_rows(0), num_cols(0), num_nonzeros(0), row_offsets(NULL), column_indices(NULL), values(NULL) 
    {
#ifdef CUB_MKL
        numa_malloc = ((numa_available() >= 0) && (numa_num_task_nodes() > 1));
#else
        numa_malloc = false;
#endif
    }


    /**
     * Clear
     */
    void Clear()
    {
#ifdef CUB_MKL
        if (numa_malloc) 
        {
            numa_free(row_offsets, sizeof(OffsetT) * (num_rows + 1));
            numa_free(values, sizeof(ValueT) * num_nonzeros);
            numa_free(column_indices, sizeof(OffsetT) * num_nonzeros);
        }
        else
        {
            if (row_offsets)    mkl_free(row_offsets);
            if (column_indices) mkl_free(column_indices);
            if (values)         mkl_free(values);
        }

#else
        if (row_offsets)    delete[] row_offsets;
        if (column_indices) delete[] column_indices;
        if (values)         delete[] values;
#endif

        row_offsets = NULL;
        column_indices = NULL;
        values = NULL;
    }

    /**
     * Destructor
     */
    ~CsrMatrix()
    {
        Clear();
    }

    GraphStats Stats()
    {
        GraphStats stats;
        stats.num_rows = num_rows;
        stats.num_cols = num_cols;
        stats.num_nonzeros = num_nonzeros;

        //
        // Compute diag-distance statistics
        //

        OffsetT samples     = 0;
        double  mean        = 0.0;
        double  ss_tot      = 0.0;

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];
                double x                = (col > row) ? col - row : row - col;

                samples++;
                double delta            = x - mean;
                mean                    = mean + (delta / samples);
                ss_tot                  += delta * (x - mean);
            }
        }
        stats.diag_dist_mean            = mean;
        double variance                 = ss_tot / samples;
        stats.diag_dist_std_dev         = sqrt(variance);


        //
        // Compute deming statistics
        //

        samples         = 0;
        double mean_x   = 0.0;
        double mean_y   = 0.0;
        double ss_x     = 0.0;
        double ss_y     = 0.0;

        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];

                samples++;
                double x                = col;
                double y                = row;
                double delta;

                delta                   = x - mean_x;
                mean_x                  = mean_x + (delta / samples);
                ss_x                    += delta * (x - mean_x);

                delta                   = y - mean_y;
                mean_y                  = mean_y + (delta / samples);
                ss_y                    += delta * (y - mean_y);
            }
        }

        samples         = 0;
        double s_xy     = 0.0;
        double s_xxy    = 0.0;
        double s_xyy    = 0.0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT nz_idx_start    = row_offsets[row];
            OffsetT nz_idx_end      = row_offsets[row + 1];

            for (int nz_idx = nz_idx_start; nz_idx < nz_idx_end; ++nz_idx)
            {
                OffsetT col             = column_indices[nz_idx];

                samples++;
                double x                = col;
                double y                = row;

                double xy =             (x - mean_x) * (y - mean_y);
                double xxy =            (x - mean_x) * (x - mean_x) * (y - mean_y);
                double xyy =            (x - mean_x) * (y - mean_y) * (y - mean_y);
                double delta;

                delta                   = xy - s_xy;
                s_xy                    = s_xy + (delta / samples);

                delta                   = xxy - s_xxy;
                s_xxy                   = s_xxy + (delta / samples);

                delta                   = xyy - s_xyy;
                s_xyy                   = s_xyy + (delta / samples);
            }
        }

        double s_xx     = ss_x / num_nonzeros;
        double s_yy     = ss_y / num_nonzeros;

        double deming_slope = (s_yy - s_xx + sqrt(((s_yy - s_xx) * (s_yy - s_xx)) + (4 * s_xy * s_xy))) / (2 * s_xy);

        stats.pearson_r = (num_nonzeros * s_xy) / (sqrt(ss_x) * sqrt(ss_y));


        //
        // Compute row-length statistics
        //

        // Sample mean
        stats.row_length_mean       = double(num_nonzeros) / num_rows;
        variance                    = 0.0;
        stats.row_length_skewness   = 0.0;
        for (OffsetT row = 0; row < num_rows; ++row)
        {
            OffsetT length              = row_offsets[row + 1] - row_offsets[row];
            double delta                = double(length) - stats.row_length_mean;
            variance   += (delta * delta);
            stats.row_length_skewness   += (delta * delta * delta);
        }
        variance                    /= num_rows;
        stats.row_length_std_dev    = sqrt(variance);
        stats.row_length_skewness   = (stats.row_length_skewness / num_rows) / pow(stats.row_length_std_dev, 3.0);
        stats.row_length_variation  = stats.row_length_std_dev / stats.row_length_mean;

        return stats;
    }

    /**
     * Build CSR matrix from sorted COO matrix
     */
    void FromCoo(const CooMatrix<ValueT, OffsetT> &coo_matrix)
    {
        num_rows        = coo_matrix.num_rows;
        num_cols        = coo_matrix.num_cols;
        num_nonzeros    = coo_matrix.num_nonzeros;

#ifdef CUB_MKL

        if (numa_malloc)
        {
            numa_set_strict(1);
//            numa_set_bind_policy(1);

//        values          = (ValueT*) numa_alloc_interleaved(sizeof(ValueT) * num_nonzeros);
//        row_offsets     = (OffsetT*) numa_alloc_interleaved(sizeof(OffsetT) * (num_rows + 1));
//        column_indices  = (OffsetT*) numa_alloc_interleaved(sizeof(OffsetT) * num_nonzeros);

            row_offsets     = (OffsetT*) numa_alloc_onnode(sizeof(OffsetT) * (num_rows + 1), 0);
            column_indices  = (OffsetT*) numa_alloc_onnode(sizeof(OffsetT) * num_nonzeros, 0);
            values          = (ValueT*) numa_alloc_onnode(sizeof(ValueT) * num_nonzeros, 1);
        }
        else
        {
            values          = (ValueT*) mkl_malloc(sizeof(ValueT) * num_nonzeros, 4096);
            row_offsets     = (OffsetT*) mkl_malloc(sizeof(OffsetT) * (num_rows + 1), 4096);
            column_indices  = (OffsetT*) mkl_malloc(sizeof(OffsetT) * num_nonzeros, 4096);

        }

#else
        row_offsets     = new OffsetT[num_rows + 1];
        column_indices  = new OffsetT[num_nonzeros];
        values          = new ValueT[num_nonzeros];
#endif

        OffsetT prev_row = -1;
        for (OffsetT current_edge = 0; current_edge < num_nonzeros; current_edge++)
        {
            OffsetT current_row = coo_matrix.coo_tuples[current_edge].row;

            // Fill in rows up to and including the current row
            for (OffsetT row = prev_row + 1; row <= current_row; row++)
            {
                row_offsets[row] = current_edge;
            }
            prev_row = current_row;

            column_indices[current_edge]    = coo_matrix.coo_tuples[current_edge].col;
            values[current_edge]            = coo_matrix.coo_tuples[current_edge].val;
        }

        // Fill out any trailing edgeless vertices (and the end-of-list element)
        for (OffsetT row = prev_row + 1; row <= num_rows; row++)
        {
            row_offsets[row] = num_nonzeros;
        }
    }


    /**
     * Display log-histogram to stdout
     */
    void DisplayHistogram()
    {
        // Initialize
        int log_counts[9];
        for (int i = 0; i < 9; i++)
        {
            log_counts[i] = 0;
        }

        // Scan
        int max_log_length = -1;
        for (OffsetT row = 0; row < num_rows; row++)
        {
            OffsetT length = row_offsets[row + 1] - row_offsets[row];

            int log_length = -1;
            while (length > 0)
            {
                length /= 10;
                log_length++;
            }
            if (log_length > max_log_length)
            {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("CSR matrix (%d rows, %d columns, %d non-zeros):\n", (int) num_rows, (int) num_cols, (int) num_nonzeros);
        for (int i = -1; i < max_log_length + 1; i++)
        {
            printf("\tDegree 1e%d: \t%d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / num_cols);
        }
        fflush(stdout);
    }


    /**
     * Display matrix to stdout
     */
    void Display()
    {
        printf("Input Matrix:\n");
        for (OffsetT row = 0; row < num_rows; row++)
        {
            printf("%d [@%d, #%d]: ", row, row_offsets[row], row_offsets[row + 1] - row_offsets[row]);
            for (OffsetT current_edge = row_offsets[row]; current_edge < row_offsets[row + 1]; current_edge++)
            {
                printf("%d (%f), ", column_indices[current_edge], values[current_edge]);
            }
            printf("\n");
        }
        fflush(stdout);
    }


};



/******************************************************************************
 * Matrix transformations
 ******************************************************************************/

// Comparator for ordering rows by degree (lowest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByLow
{
    OffsetT* row_degrees;
    OrderByLow(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] < row_degrees[b])
            return true;
        else if (row_degrees[a] > row_degrees[b])
            return false;
        else
            return (a < b);
    }
};

// Comparator for ordering rows by degree (highest first), then by row-id (lowest first)
template <typename OffsetT>
struct OrderByHigh
{
    OffsetT* row_degrees;
    OrderByHigh(OffsetT* row_degrees) : row_degrees(row_degrees) {}

    bool operator()(const OffsetT &a, const OffsetT &b)
    {
        if (row_degrees[a] > row_degrees[b])
            return true;
        else if (row_degrees[a] < row_degrees[b])
            return false;
        else
            return (a < b);
    }
};



/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmRelabel(
    CsrMatrix<ValueT, OffsetT>&     matrix,
    OffsetT*                        relabel_indices)
{
    // Initialize row degrees
    OffsetT* row_degrees_in     = new OffsetT[matrix.num_rows];
    OffsetT* row_degrees_out    = new OffsetT[matrix.num_rows];
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        row_degrees_in[row]         = 0;
        row_degrees_out[row]        = matrix.row_offsets[row + 1] - matrix.row_offsets[row];
    }
    for (OffsetT nonzero = 0; nonzero < matrix.num_nonzeros; ++nonzero)
    {
        row_degrees_in[matrix.column_indices[nonzero]]++;
    }

    // Initialize unlabeled set 
    typedef std::set<OffsetT, OrderByLow<OffsetT> > UnlabeledSet;
    typename UnlabeledSet::key_compare  unlabeled_comp(row_degrees_in);
    UnlabeledSet                        unlabeled(unlabeled_comp);
    for (OffsetT row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row]    = -1;
        unlabeled.insert(row);
    }

    // Initialize queue set
    std::deque<OffsetT> q;

    // Process unlabeled vertices (traverse connected components)
    OffsetT relabel_idx = 0;
    while (!unlabeled.empty())
    {
        // Seed the unvisited frontier queue with the unlabeled vertex of lowest-degree
        OffsetT vertex = *unlabeled.begin();
        q.push_back(vertex);

        while (!q.empty())
        {
            vertex = q.front();
            q.pop_front();

            if (relabel_indices[vertex] == -1)
            {
                // Update this vertex
                unlabeled.erase(vertex);
                relabel_indices[vertex] = relabel_idx;
                relabel_idx++;

                // Sort neighbors by degree
                OrderByLow<OffsetT> neighbor_comp(row_degrees_in);
                std::sort(
                    matrix.column_indices + matrix.row_offsets[vertex],
                    matrix.column_indices + matrix.row_offsets[vertex + 1],
                    neighbor_comp);

                // Inspect neighbors, adding to the out frontier if unlabeled
                for (OffsetT neighbor_idx = matrix.row_offsets[vertex];
                    neighbor_idx < matrix.row_offsets[vertex + 1];
                    ++neighbor_idx)
                {
                    OffsetT neighbor = matrix.column_indices[neighbor_idx];
                    q.push_back(neighbor);
                }
            }
        }
    }

/*
    // Reverse labels
    for (int row = 0; row < matrix.num_rows; ++row)
    {
        relabel_indices[row] = matrix.num_rows - relabel_indices[row] - 1;
    }
*/

    // Cleanup
    if (row_degrees_in) delete[] row_degrees_in;
    if (row_degrees_out) delete[] row_degrees_out;
}


/**
 * Reverse Cuthill-McKee
 */
template <typename ValueT, typename OffsetT>
void RcmRelabel(
    CsrMatrix<ValueT, OffsetT>&     matrix,
    bool                            verbose = false)
{
    // Do not process if not square
    if (matrix.num_cols != matrix.num_rows)
    {
        if (verbose) {
            printf("RCM transformation ignored (not square)\n"); fflush(stdout);
        }
        return;
    }

    // Initialize relabel indices
    OffsetT* relabel_indices = new OffsetT[matrix.num_rows];

    if (verbose) {
        printf("RCM relabeling... "); fflush(stdout);
    }

    RcmRelabel(matrix, relabel_indices);

    if (verbose) {
        printf("done. Reconstituting... "); fflush(stdout);
    }

    // Create a COO matrix from the relabel indices
    CooMatrix<ValueT, OffsetT> coo_matrix;
    coo_matrix.InitCsrRelabel(matrix, relabel_indices);

    // Reconstitute the CSR matrix from the sorted COO tuples
    if (relabel_indices) delete[] relabel_indices;
    matrix.Clear();
    matrix.FromCoo(coo_matrix);

    if (verbose) {
        printf("done. "); fflush(stdout);
    }
}




