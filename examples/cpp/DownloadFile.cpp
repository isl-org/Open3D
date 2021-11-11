#include <curl/curl.h>

#include <iostream>
#include <string>

static size_t WriteCallback(void *contents,
                            size_t size,
                            size_t nmemb,
                            void *userp) {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

int main(void) {
    std::cout << "testing 1";
    CURL *curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {

        std::cout << "testing 3";

        curl_easy_setopt(curl, CURLOPT_URL, "http://www.google.com");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        (void)res;
        curl_easy_cleanup(curl);

        std::cout << readBuffer << std::endl;
    }

    std::cout << "testing 2";

    return 0;
}
