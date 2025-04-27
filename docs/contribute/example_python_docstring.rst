Single line short description: Example docstring for Open3D.

Use the ``Args``, ``Returns``, ``Example``, ``Note`` sections to describe classes
or functions as needed.
Giving a code example showing how to use a class or function in the ``Example``
section is highly recommended.

To explain what a method does you can use code and math.
This is a literal block started with ``::`` followed by a blank line to show code::
    
    pip install open3d 

This is an inline equation :math:`e^{i\pi} + 1 = 0` and this is a display equation
(don't forget the blank line before the ``..math::`` directive):

.. math::
    e^{i\pi} + 1 = 0.


The default alignment for multiple equations is right-align, which is often not 
desired. 
To align multiple equations manually use the ``align`` environment with ``:nowrap:``
**(don't forget to add a blank line after :nowrap: to make this work!)**

.. math::
    :nowrap:

    \begin{align}
        x &= r \sin(\theta) \cos(\phi) \\
        y &= r \sin(\theta) \sin(\phi) \\
        z &= r \cos(\theta).
    \end{align}


Note:
    You can use inline markup to format your documentation.

    - *italics* for emphasis
    - **boldface** for strong emphasis
    - ``code`` for inline code

    A list must be separated with blank lines to work.
        
    - This is a list
    - Second item

      - A nested list must be ..
      - separated with blank lines too
      - **Use only two spaces to indent or it will be treated as a block quote.**

    - Third item

    This is a link `The space before the pointy left bracket is important! <https://www.open3d.org>`_
    This does not work, `you forgot the space<https://www.open3d.org>`_


Args:
    param1 (o3d.core.Tensor): Specify the shape with round brackets, (N,3), and 
        the dtype, Float32, as necessary for Tensor parameters

    param2 (str): Another parameter

Returns:
    Returned values can be tuples, lists or dictionaries sometimes.
    Describing every item can be done with lists

    - o3d.core.Tensor: vertices with shape (N,3) and type Float32.
    - int: the number of vertices. This is N.

    or definition lists, which is recommended if the returned value is a dictionary

    vertices (o3d.core.Tensor) [term must be a single line]
        Description, first paragraph.
        A tensor with the vertices of shape (N,3) and type Float32.

        Description, second paragraph.
        The tensor has the same shape as param1.        

    count (int)
        Description.
        The number of vertices. This is N.


Example:
    This is a code example showing how Open3D is imported::

        import open3d as o3d