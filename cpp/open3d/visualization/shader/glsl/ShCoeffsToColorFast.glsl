// Refering to [this link](https://github.com/nerfstudio-project/gsplat/blob/bd64a47414e182dc105c1a2fdb6691068518d060/gsplat/cuda/include/spherical_harmonics.cuh)

vec3 sh0_coeffs_to_color_fast(vec3 coeffs_sh0) {
    return 0.2820947917738781 * coeffs_sh0;
}

// vec3 dir must be normalized vector
vec3 sh1_coeffs_to_color_fast(vec3 dir, float coeffs[3 * 3]) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        result[i] = 0.48860251190292 * (-dir.y * coeffs[0 + i] + dir.z * coeffs[1 * 3 + i] - dir.x * coeffs[2 * 3 + i]);
    }
    return result;
}

// vec3 dir must be normalized vector
vec3 sh2_coeffs_to_color_fast(vec3 dir, float coeffs[5 * 3]) {
    float inorm = inversesqrt(dot(dir, dir));
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;

    float z2 = dir.z * dir.z;
    float fTmp0B = -1.092548430592079 * dir.z;
    float fC1 = dir.x * dir.x - dir.y * dir.y;
    float fS1 = 2.0 * dir.x * dir.y;

    float pSH6 = (0.9461746957575601 * z2 - 0.3153915652525201);
    float pSH7 = fTmp0B * dir.x;
    float pSH5 = fTmp0B * dir.y;
    float pSH8 = 0.5462742152960395 * fC1;
    float pSH4 = 0.5462742152960395 * fS1;

    vec3 result;
    for (int i = 0; i < 3; i++) {
        result[i] = pSH4 * coeffs[0 + i] + pSH5 * coeffs[1 * 3 + i] +
        pSH6 * coeffs[2 * 3 + i] + pSH7 * coeffs[3 * 3 + i] +
        pSH8 * coeffs[4 * 3 + i];
    }
    
    return result;
}