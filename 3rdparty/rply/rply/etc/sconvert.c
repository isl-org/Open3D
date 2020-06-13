#include <stdio.h> 
#include "rply.h"

static int callback(p_ply_argument argument) {
    void *pdata;
    /* just pass the value from the input file to the output file */
    ply_get_argument_user_data(argument, &pdata, NULL);
    ply_write((p_ply) pdata, ply_get_argument_value(argument));
    return 1;
}

static int setup_callbacks(p_ply iply, p_ply oply) {
    p_ply_element element = NULL;
    /* iterate over all elements in input file */
    while ((element = ply_get_next_element(iply, element))) {
        p_ply_property property = NULL;
        long ninstances = 0;
        const char *element_name;
        ply_get_element_info(element, &element_name, &ninstances);
        /* add this element to output file */
        if (!ply_add_element(oply, element_name, ninstances)) return 0;
        /* iterate over all properties of current element */
        while ((property = ply_get_next_property(element, property))) {
            const char *property_name;
            e_ply_type type, length_type, value_type;
            ply_get_property_info(property, &property_name, &type, 
                    &length_type, &value_type);
            /* setup input callback for this property */
            ply_set_read_cb(iply, element_name, property_name, callback, 
                oply, 0);
            /* add this property to output file */
            if (!ply_add_property(oply, property_name, type, length_type, 
                    value_type)) return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    const char *value;
    p_ply iply, oply; 
    (void) argc; (void) argv;
    iply = ply_open("input.ply", NULL, 0, NULL);
    if (!iply) return 1; 
    if (!ply_read_header(iply)) return 1; 
    oply = ply_create("output.ply", PLY_LITTLE_ENDIAN, NULL, 0, NULL);
    if (!oply) return 1;
    if (!setup_callbacks(iply, oply)) return 1; 
    /* pass comments and obj_infos from input to output */
    value = NULL;
    while ((value = ply_get_next_comment(iply, value)))
        if (!ply_add_comment(oply, value)) return 1; 
    value = NULL;
    while ((value = ply_get_next_obj_info(iply, value)))
        if (!ply_add_obj_info(oply, value)) return 1;;
    /* write output header */
    if (!ply_write_header(oply)) return 1; 
    /* read input file generating callbacks that pass data to output file */
    if (!ply_read(iply)) return 1; 
    /* close up, we are done */
    if (!ply_close(iply)) return 1; 
    if (!ply_close(oply)) return 1;
    return 0;
}
