#define CURL_STATICLIB
#include <curl/curl.h>
#include <stdio.h>
// #include <curl/types.h>
#include <curl/easy.h>
#include <stdlib.h>
#include <string.h>

#define false 0

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}

int main(void) {
    CURL *curl;
    FILE *fp;
    CURLcode res;

    const char url[] =
            "https://github.com/reyanshsolis/rey_download/releases/download/"
            "test_data/test_file.zip";
    const char outfilename[FILENAME_MAX] = "test_file.zip";

    curl_version_info_data *vinfo = curl_version_info(CURLVERSION_NOW);

    if (vinfo->features & CURL_VERSION_SSL) {
        printf("CURL: SSL enabled\n");
    } else {
        printf("CURL: SSL not enabled\n");
    }

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(outfilename, "wb");

        /* Setup the https:// verification options. Note we   */
        /* do this on all requests as there may be a redirect */
        /* from http to https and we still want to verify     */
        curl_easy_setopt(curl, CURLOPT_URL, url);

        // Follow redirection in link.
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // curl_easy_setopt(curl, CURLOPT_CAINFO, "./ca-bundle.crt");

        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, false);

        // Write function callback.
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);

        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        // curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1); //Prevent "longjmp
        // causes uninitialized stack frame" bug curl_easy_setopt(curl,
        // CURLOPT_ACCEPT_ENCODING, "deflate"); std::stringstream out;
        // curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        // curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);

        // Performs the request. res gets the retun code.
        res = curl_easy_perform(curl);
        (void)res;

        // Cleanup.
        curl_easy_cleanup(curl);
        // Close file.
        fclose(fp);
    }
    return 0;
}
