
#include <cassert>
#include <cmath>

#include "Vector3.h"

#include <iomanip>
using namespace std;

// ----------------------------------------------------------------------------
// Display an open3d::Vector3<T>::Type.
// ----------------------------------------------------------------------------
ostream& open3d::operator <<(ostream& os, const open3d::Vector3d& v)
{
    // save
    ios_base::fmtflags flags = cout.flags();

    // set new precision
    cout.precision(DBL_PRECISION);
    cout.setf(ios::fixed);

    cout << setw(DBL_WIDTH) << v.x;
    cout << setw(DBL_WIDTH) << v.y;
    cout << setw(DBL_WIDTH) << v.z;

    // restore
    cout.flags(flags);

    cout.flush();

    return os;
}
ostream& open3d::operator <<(ostream& os, const open3d::Vector3f& v)
{
    // save
    ios_base::fmtflags flags = cout.flags();

    // set new precision
    cout.precision(FLT_PRECISION);
    cout.setf(ios::fixed);

    cout << setw(FLT_WIDTH) << v.x;
    cout << setw(FLT_WIDTH) << v.y;
    cout << setw(FLT_WIDTH) << v.z;

    // restore
    cout.flags(flags);

    cout.flush();

    return os;
}
ostream& open3d::operator <<(ostream& os, const open3d::Vector3i& v)
{
    cout << setw(FLT_WIDTH) << v.x;
    cout << setw(FLT_WIDTH) << v.y;
    cout << setw(FLT_WIDTH) << v.z;

    cout.flush();

    return os;
}
