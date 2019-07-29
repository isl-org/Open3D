// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CMDPARSER_H
#define CMDPARSER_H

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace CmdParser {
typedef std::function<void(const std::vector<char *> &)> OptionHandlerFunc;
typedef std::function<void()> VoidHandlerFunc;

class ArgumentError : public std::runtime_error {
public:
    ArgumentError(const std::string option, const std::string message)
        : std::runtime_error(message), m_option(option), m_message(message) {}

    const std::string option() const { return m_option; }

private:
    const std::string m_option, m_message;
};

class OptionParser {
public:
    OptionParser() {}

    void RegisterOption(std::string option_list,
                        std::string description,
                        int arg_count,
                        OptionHandlerFunc &&handler) {
        std::istringstream split(option_list);
        std::ostringstream options_str;
        std::string option;
        bool first = true;
        while (std::getline(split, option, '|')) {
            m_cmd_options[option] = OptionHandler{arg_count, handler};

            if (first) {
                options_str << option;
                first = false;
            } else {
                options_str << ", " << option;
            }
        }
        m_cmd_usage.push_back(std::make_pair(options_str.str(), description));
    }

    void RegisterOption(std::string option_list,
                        std::string description,
                        VoidHandlerFunc &&handler) {
        RegisterOption(option_list, description, 0,
                       [handler](const std::vector<char *> &) { handler(); });
    }

    void PrintOptions() {
        std::cout << " Options:" << std::endl;
        size_t max_options_length = 0;
        for (std::pair<std::string, std::string> usage : m_cmd_usage) {
            if (usage.first.length() > max_options_length)
                max_options_length = usage.first.length();
        }
        for (std::pair<std::string, std::string> usage : m_cmd_usage) {
            std::cout << "  " << usage.first
                      << std::string(
                                 max_options_length - usage.first.length() + 2,
                                 ' ');
            std::istringstream split(usage.second);
            std::string line;
            bool first = true;
            while (std::getline(split, line)) {
                if (first) {
                    std::cout << line << std::endl;
                    first = false;
                } else {
                    std::cout << "  "
                              << std::string(max_options_length + 4, ' ')
                              << line << std::endl;
                }
            }
        }
    }

    /** Parse the command line arguments, and run the handler for each.
     *
     * Returns the number of remaining arguments after parsing.
     * May throw ArgumentError if a handler fails.
     */
    int ParseCmd(int argc, char **argv) {
        if (argc < 2) return 0;

        for (int i = 1; i < argc; i++) {
            std::string option = argv[i];
            if (m_cmd_options.count(option) == 0) return argc - i;

            const OptionHandler &handler = m_cmd_options[option];
            std::vector<char *> args;
            if (i + handler.arg_count < argc) {
                for (int j = 0; j < handler.arg_count; j++) {
                    args.push_back(argv[i + j + 1]);
                }
                try {
                    handler.handler(args);
                } catch (ArgumentError &) {
                    throw;
                } catch (std::exception &e) {
                    throw ArgumentError(option, e.what());
                }
                i += handler.arg_count;
            } else {
                return argc - i - 1;
            }
        }
        return 0;
    }

private:
    struct OptionHandler {
        int arg_count;
        OptionHandlerFunc handler;
    };

    std::map<std::string, OptionHandler> m_cmd_options;
    std::vector<std::pair<std::string, std::string>> m_cmd_usage;
};
}  // namespace CmdParser

#endif /* CMDPARSER_H */
