.. _poux_book:

3D Data Science in Python featuring Open3D
==========================================

This book is a great introduction to modern 3D data processing and contains a
wonderful collection of techniques from Open3D and other 3D software.

.. image:: https://learning.oreilly.com/covers/urn:orm:book:9781098161323/400w/
    :alt: Book cover
    :width: 400px
    :align: center
    :target: https://learning.oreilly.com/library/view/3d-data-science/9781098161323/


About the book
--------------

Our physical world is grounded in three dimensions. To create technology that
can reason about and interact with it, our data must be 3D too. This practical
guide offers data scientists, engineers, and researchers a hands-on approach to
working with 3D data using Python. From 3D reconstruction to 3D deep learning
techniques, you'll learn how to extract valuable insights from massive datasets,
including point clouds, voxels, 3D CAD models, meshes, images, and more.

Dr. Florent Poux helps you leverage the potential of cutting-edge algorithms and
spatial AI models to develop production-ready systems with a focus on
automation. You'll get the 3D data science knowledge and code to:

* Understand core concepts and representations of 3D data
* Load, manipulate, analyze, and visualize 3D data using powerful Python libraries
* Apply advanced AI algorithms for 3D pattern recognition (supervised and unsupervised)
* Use 3D reconstruction techniques to generate 3D datasets
* Implement automated 3D modeling and generative AI workflows
* Explore practical applications in areas like computer vision/graphics, geospatial intelligence, scientific computing, robotics, and autonomous driving
* Build accurate digital environments that spatial AI solutions can leverage

Florent Poux is an esteemed authority in the field of 3D data science who
teaches and conducts research for top European universities. He's also head
professor at the 3D Geodata Academy and innovation director for French Tech 120
companies.

Here is a sample Open3D code snippet and visualization from Chapter 5 of the book:

.. code-block:: python

   # Selecting points by distance from POI (your house) using the k-d tree 
   [k, idx, _] = pcd_tree.search_radius_vector_3d(POI, dist_POI)
   pcd_selection = pcd_o3d.select_by_index(idx)
   o3d.visualization.draw_geometries([pcd_selection, mesh])


.. figure:: https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098161323/files/assets/ddsp_0508.png
    :alt: Selecting points by distance from POI using k-d tree
    :width: 400px
    :align: center

    Selecting points by distance from POI (your house) using the k-d tree