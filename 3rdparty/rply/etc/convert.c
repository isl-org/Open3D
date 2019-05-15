#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdarg.h> 
#include "rply.h"

/* internal function prototypes */
static void error(const char *fmt, ...);
static void help(void);
static void parse_arguments(int argc, char **argv, 
        e_ply_storage_mode *storage_mode, 
        const char **iname, const char **oname);
static void setup_callbacks(p_ply iply, p_ply oply);

/* given a format mode, an input file name and an output file name,
 * convert input file to output in given format mode */
int main(int argc, char **argv) {
    const char *value = NULL;
    e_ply_storage_mode storage_mode = PLY_LITTLE_ENDIAN;
    const char *iname = NULL, *oname = NULL;
    p_ply iply = NULL, oply = NULL;
    /* parse command line arguments */
    parse_arguments(argc, argv, &storage_mode, &iname, &oname);
    /* open input file and make sure we parsed its header */
    iply = ply_open(iname, NULL, 0, NULL);
    if (!iply) error("Unable to open file '%s'", iname);
    if (!ply_read_header(iply)) error("Failed reading '%s' header", iname);
    /* create output file */
    oply = ply_create(oname, storage_mode, NULL, 0, NULL);
    if (!oply) error("Unable to create file '%s'", oname);
    /* create elements and properties in output file and 
     * setup callbacks for them in input file */
    setup_callbacks(iply, oply); 
    /* pass comments and obj_infos from input to output */
    value = NULL;
    while ((value = ply_get_next_comment(iply, value)))
        if (!ply_add_comment(oply, value))
            error("Failed adding comments");
    value = NULL;
    while ((value = ply_get_next_obj_info(iply, value)))
        if (!ply_add_obj_info(oply, value))
            error("Failed adding comments");
    /* write output header */
    if (!ply_write_header(oply)) error("Failed writing '%s' header", oname);
    /* read input file generating callbacks that pass data to output file */
    if (!ply_read(iply)) error("Conversion failed");
    /* close up, we are done */
    if (!ply_close(iply)) error("Error closing file '%s'", iname);
    if (!ply_close(oply)) error("Error closing file '%s'", oname);
    return 0;
}

/* prints an error message and exits */
static void error(const char *fmt, ...) {   
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}               

/* prints the help message and exits */
static void help(void) {
    error("Usage:\n"  
            "    convert <option> <input> <output>\n"
            "Options:\n"
            "    -a, --ascii: convert to ascii format\n"
            "    -b, --big-endian: convert to big-endian format\n"
            "    -l, --little-endian: convert to little-endian format\n");
}

/* parse command line parameters */
static void parse_arguments(int argc, char **argv, 
        e_ply_storage_mode *storage_mode, 
        const char **iname, const char **oname) {
    if (argc < 4) help();
    if (strcmp(argv[1], "--ascii") == 0 || 
            strcmp(argv[1], "-a") == 0)
        *storage_mode = PLY_ASCII;
    else if (strcmp(argv[1], "--little-endian") == 0 || 
            strcmp(argv[1], "-l") == 0) 
        *storage_mode = PLY_LITTLE_ENDIAN;
    else if (strcmp(argv[1], "--big-endian") == 0 || 
            strcmp(argv[1], "-b") == 0) 
        *storage_mode = PLY_BIG_ENDIAN;
    else help(); 
    *iname = argv[2];
    *oname = argv[3];
}

/* read callback */
static int callback(p_ply_argument argument) {
    void *pdata;
    /* just pass the value from the input file to the output file */
    ply_get_argument_user_data(argument, &pdata, NULL);
    ply_write((p_ply) pdata, ply_get_argument_value(argument));
    return 1;
}

/* prepares the conversion */
static void setup_callbacks(p_ply iply, p_ply oply) {
    p_ply_element element = NULL;
    /* iterate over all elements in input file */
    while ((element = ply_get_next_element(iply, element))) {
        p_ply_property property = NULL;
        long ninstances = 0;
        const char *element_name;
        ply_get_element_info(element, &element_name, &ninstances);
        /* add this element to output file */
        if (!ply_add_element(oply, element_name, ninstances))
            error("Unable to add output element '%s'", element_name);
        /* iterate over all properties of current element */
        while ((property = ply_get_next_property(element, property))) {
            const char *property_name;
            e_ply_type type, length_type, value_type;
            ply_get_property_info(property, &property_name, &type, 
                    &length_type, &value_type);
            /* setup input callback for this property */
            if (!ply_set_read_cb(iply, element_name, property_name, callback, 
                    oply, 0))
                error("Unable to setup input callback for property '%s'", 
                        property_name);
            /* add this property to output file */
            if (!ply_add_property(oply, property_name, type, length_type, 
                    value_type))
                error("Unable to add output property '%s'", property_name);
        }
    }
}
