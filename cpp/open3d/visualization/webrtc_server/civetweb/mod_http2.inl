/* Prototype implementation for HTTP2. Do not use in production.
 * There may be memory leaks, security vulnerabilities, ...
 */


/***********************************************************************/
/***  HPACK                                                          ***/
/***********************************************************************/

/* hpack predefined table. See:
 * https://tools.ietf.org/html/rfc7541#appendix-A
 */

static struct mg_header hpack_predefined[62] = {{NULL, NULL},
                                                {":authority", NULL},
                                                {":method", "GET"},
                                                {":method", "POST"},
                                                {":path", "/"},
                                                {":path", "/index.html"},
                                                {":scheme", "http"},
                                                {":scheme", "https"},
                                                {":status", "200"},
                                                {":status", "204"},
                                                {":status", "206"},
                                                {":status", "304"},
                                                {":status", "400"},
                                                {":status", "404"},
                                                {":status", "500"},
                                                {"accept-charset", NULL},
                                                {"accept-encoding", NULL},
                                                {"accept-language", NULL},
                                                {"accept-ranges", NULL},
                                                {"accept", NULL},
                                                {"access-control-allow-origin",
                                                 NULL},
                                                {"age", NULL},
                                                {"allow", NULL},
                                                {"authorization", NULL},
                                                {"cache-control", NULL},
                                                {"content-encoding", NULL},
                                                {"content-disposition", NULL},
                                                {"content-language", NULL},
                                                {"content-length", NULL},
                                                {"content-location", NULL},
                                                {"content-range", NULL},
                                                {"content-type", NULL},
                                                {"cookie", NULL},
                                                {"date", NULL},
                                                {"etag", NULL},
                                                {"expect", NULL},
                                                {"expires", NULL},
                                                {"from", NULL},
                                                {"host", NULL},
                                                {"if-match", NULL},
                                                {"if-modified-since", NULL},
                                                {"if-none-match", NULL},
                                                {"if-range", NULL},
                                                {"if-unmodified-since", NULL},
                                                {"last-modified", NULL},
                                                {"link", NULL},
                                                {"location", NULL},
                                                {"max-forwards", NULL},
                                                {"proxy-authenticate", NULL},
                                                {"proxy-authorization", NULL},
                                                {"range", NULL},
                                                {"referer", NULL},
                                                {"refresh", NULL},
                                                {"retry-after", NULL},
                                                {"server", NULL},
                                                {"set-cookie", NULL},
                                                {"strict-transport-security",
                                                 NULL},
                                                {"transfer-encoding", NULL},
                                                {"user-agent", NULL},
                                                {"vary", NULL},
                                                {"via", NULL},
                                                {"www-authenticate", NULL}};

/* Huffman decoding: https://tools.ietf.org/html/rfc7541#appendix-B

'0' ( 48)  |00000                                         0  [ 5]
'1' ( 49)  |00001                                         1  [ 5]
'2' ( 50)  |00010                                         2  [ 5]
'a' ( 97)  |00011                                         3  [ 5]
'c' ( 99)  |00100                                         4  [ 5]
'e' (101)  |00101                                         5  [ 5]
'i' (105)  |00110                                         6  [ 5]
'o' (111)  |00111                                         7  [ 5]
's' (115)  |01000                                         8  [ 5]
't' (116)  |01001                                         9  [ 5]
' ' ( 32)  |010100                                       14  [ 6]
'%' ( 37)  |010101                                       15  [ 6]
'-' ( 45)  |010110                                       16  [ 6]
'.' ( 46)  |010111                                       17  [ 6]
'/' ( 47)  |011000                                       18  [ 6]
'3' ( 51)  |011001                                       19  [ 6]
'4' ( 52)  |011010                                       1a  [ 6]
'5' ( 53)  |011011                                       1b  [ 6]
'6' ( 54)  |011100                                       1c  [ 6]
'7' ( 55)  |011101                                       1d  [ 6]
'8' ( 56)  |011110                                       1e  [ 6]
'9' ( 57)  |011111                                       1f  [ 6]
'=' ( 61)  |100000                                       20  [ 6]
'A' ( 65)  |100001                                       21  [ 6]
'_' ( 95)  |100010                                       22  [ 6]
'b' ( 98)  |100011                                       23  [ 6]
'd' (100)  |100100                                       24  [ 6]
'f' (102)  |100101                                       25  [ 6]
'g' (103)  |100110                                       26  [ 6]
'h' (104)  |100111                                       27  [ 6]
'l' (108)  |101000                                       28  [ 6]
'm' (109)  |101001                                       29  [ 6]
'n' (110)  |101010                                       2a  [ 6]
'p' (112)  |101011                                       2b  [ 6]
'r' (114)  |101100                                       2c  [ 6]
'u' (117)  |101101                                       2d  [ 6]
':' ( 58)  |1011100                                      5c  [ 7]
'B' ( 66)  |1011101                                      5d  [ 7]
'C' ( 67)  |1011110                                      5e  [ 7]
'D' ( 68)  |1011111                                      5f  [ 7]
'E' ( 69)  |1100000                                      60  [ 7]
'F' ( 70)  |1100001                                      61  [ 7]
'G' ( 71)  |1100010                                      62  [ 7]
'H' ( 72)  |1100011                                      63  [ 7]
'I' ( 73)  |1100100                                      64  [ 7]
'J' ( 74)  |1100101                                      65  [ 7]
'K' ( 75)  |1100110                                      66  [ 7]
'L' ( 76)  |1100111                                      67  [ 7]
'M' ( 77)  |1101000                                      68  [ 7]
'N' ( 78)  |1101001                                      69  [ 7]
'O' ( 79)  |1101010                                      6a  [ 7]
'P' ( 80)  |1101011                                      6b  [ 7]
'Q' ( 81)  |1101100                                      6c  [ 7]
'R' ( 82)  |1101101                                      6d  [ 7]
'S' ( 83)  |1101110                                      6e  [ 7]
'T' ( 84)  |1101111                                      6f  [ 7]
'U' ( 85)  |1110000                                      70  [ 7]
'V' ( 86)  |1110001                                      71  [ 7]
'W' ( 87)  |1110010                                      72  [ 7]
'Y' ( 89)  |1110011                                      73  [ 7]
'j' (106)  |1110100                                      74  [ 7]
'k' (107)  |1110101                                      75  [ 7]
'q' (113)  |1110110                                      76  [ 7]
'v' (118)  |1110111                                      77  [ 7]
'w' (119)  |1111000                                      78  [ 7]
'x' (120)  |1111001                                      79  [ 7]
'y' (121)  |1111010                                      7a  [ 7]
'z' (122)  |1111011                                      7b  [ 7]
'&' ( 38)  |11111000                                     f8  [ 8]
'*' ( 42)  |11111001                                     f9  [ 8]
',' ( 44)  |11111010                                     fa  [ 8]
';' ( 59)  |11111011                                     fb  [ 8]
'X' ( 88)  |11111100                                     fc  [ 8]
'Z' ( 90)  |11111101                                     fd  [ 8]
'!' ( 33)  |11111110|00                                 3f8  [10]
'"' ( 34)  |11111110|01                                 3f9  [10]
'(' ( 40)  |11111110|10                                 3fa  [10]
')' ( 41)  |11111110|11                                 3fb  [10]
'?' ( 63)  |11111111|00                                 3fc  [10]
''' ( 39)  |11111111|010                                7fa  [11]
'+' ( 43)  |11111111|011                                7fb  [11]
'|' (124)  |11111111|100                                7fc  [11]
'#' ( 35)  |11111111|1010                               ffa  [12]
'>' ( 62)  |11111111|1011                               ffb  [12]
    (  0)  |11111111|11000                             1ff8  [13]
'$' ( 36)  |11111111|11001                             1ff9  [13]
'@' ( 64)  |11111111|11010                             1ffa  [13]
'[' ( 91)  |11111111|11011                             1ffb  [13]
']' ( 93)  |11111111|11100                             1ffc  [13]
'~' (126)  |11111111|11101                             1ffd  [13]
'^' ( 94)  |11111111|111100                            3ffc  [14]
'}' (125)  |11111111|111101                            3ffd  [14]
'<' ( 60)  |11111111|1111100                           7ffc  [15]
'`' ( 96)  |11111111|1111101                           7ffd  [15]
'{' (123)  |11111111|1111110                           7ffe  [15]
'\' ( 92)  |11111111|11111110|000                     7fff0  [19]
    (195)  |11111111|11111110|001                     7fff1  [19]
    (208)  |11111111|11111110|010                     7fff2  [19]
    (128)  |11111111|11111110|0110                    fffe6  [20]
    (130)  |11111111|11111110|0111                    fffe7  [20]
    (131)  |11111111|11111110|1000                    fffe8  [20]
    (162)  |11111111|11111110|1001                    fffe9  [20]
    (184)  |11111111|11111110|1010                    fffea  [20]
    (194)  |11111111|11111110|1011                    fffeb  [20]
    (224)  |11111111|11111110|1100                    fffec  [20]
    (226)  |11111111|11111110|1101                    fffed  [20]
    (153)  |11111111|11111110|11100                  1fffdc  [21]
    (161)  |11111111|11111110|11101                  1fffdd  [21]
    (167)  |11111111|11111110|11110                  1fffde  [21]
    (172)  |11111111|11111110|11111                  1fffdf  [21]
    (176)  |11111111|11111111|00000                  1fffe0  [21]
    (177)  |11111111|11111111|00001                  1fffe1  [21]
    (179)  |11111111|11111111|00010                  1fffe2  [21]
    (209)  |11111111|11111111|00011                  1fffe3  [21]
    (216)  |11111111|11111111|00100                  1fffe4  [21]
    (217)  |11111111|11111111|00101                  1fffe5  [21]
    (227)  |11111111|11111111|00110                  1fffe6  [21]
    (229)  |11111111|11111111|00111                  1fffe7  [21]
    (230)  |11111111|11111111|01000                  1fffe8  [21]
    (129)  |11111111|11111111|010010                 3fffd2  [22]
    (132)  |11111111|11111111|010011                 3fffd3  [22]
    (133)  |11111111|11111111|010100                 3fffd4  [22]
    (134)  |11111111|11111111|010101                 3fffd5  [22]
    (136)  |11111111|11111111|010110                 3fffd6  [22]
    (146)  |11111111|11111111|010111                 3fffd7  [22]
    (154)  |11111111|11111111|011000                 3fffd8  [22]
    (156)  |11111111|11111111|011001                 3fffd9  [22]
    (160)  |11111111|11111111|011010                 3fffda  [22]
    (163)  |11111111|11111111|011011                 3fffdb  [22]
    (164)  |11111111|11111111|011100                 3fffdc  [22]
    (169)  |11111111|11111111|011101                 3fffdd  [22]
    (170)  |11111111|11111111|011110                 3fffde  [22]
    (173)  |11111111|11111111|011111                 3fffdf  [22]
    (178)  |11111111|11111111|100000                 3fffe0  [22]
    (181)  |11111111|11111111|100001                 3fffe1  [22]
    (185)  |11111111|11111111|100010                 3fffe2  [22]
    (186)  |11111111|11111111|100011                 3fffe3  [22]
    (187)  |11111111|11111111|100100                 3fffe4  [22]
    (189)  |11111111|11111111|100101                 3fffe5  [22]
    (190)  |11111111|11111111|100110                 3fffe6  [22]
    (196)  |11111111|11111111|100111                 3fffe7  [22]
    (198)  |11111111|11111111|101000                 3fffe8  [22]
    (228)  |11111111|11111111|101001                 3fffe9  [22]
    (232)  |11111111|11111111|101010                 3fffea  [22]
    (233)  |11111111|11111111|101011                 3fffeb  [22]
    (  1)  |11111111|11111111|1011000                7fffd8  [23]
    (135)  |11111111|11111111|1011001                7fffd9  [23]
    (137)  |11111111|11111111|1011010                7fffda  [23]
    (138)  |11111111|11111111|1011011                7fffdb  [23]
    (139)  |11111111|11111111|1011100                7fffdc  [23]
    (140)  |11111111|11111111|1011101                7fffdd  [23]
    (141)  |11111111|11111111|1011110                7fffde  [23]
    (143)  |11111111|11111111|1011111                7fffdf  [23]
    (147)  |11111111|11111111|1100000                7fffe0  [23]
    (149)  |11111111|11111111|1100001                7fffe1  [23]
    (150)  |11111111|11111111|1100010                7fffe2  [23]
    (151)  |11111111|11111111|1100011                7fffe3  [23]
    (152)  |11111111|11111111|1100100                7fffe4  [23]
    (155)  |11111111|11111111|1100101                7fffe5  [23]
    (157)  |11111111|11111111|1100110                7fffe6  [23]
    (158)  |11111111|11111111|1100111                7fffe7  [23]
    (165)  |11111111|11111111|1101000                7fffe8  [23]
    (166)  |11111111|11111111|1101001                7fffe9  [23]
    (168)  |11111111|11111111|1101010                7fffea  [23]
    (174)  |11111111|11111111|1101011                7fffeb  [23]
    (175)  |11111111|11111111|1101100                7fffec  [23]
    (180)  |11111111|11111111|1101101                7fffed  [23]
    (182)  |11111111|11111111|1101110                7fffee  [23]
    (183)  |11111111|11111111|1101111                7fffef  [23]
    (188)  |11111111|11111111|1110000                7ffff0  [23]
    (191)  |11111111|11111111|1110001                7ffff1  [23]
    (197)  |11111111|11111111|1110010                7ffff2  [23]
    (231)  |11111111|11111111|1110011                7ffff3  [23]
    (239)  |11111111|11111111|1110100                7ffff4  [23]
    (  9)  |11111111|11111111|11101010               ffffea  [24]
    (142)  |11111111|11111111|11101011               ffffeb  [24]
    (144)  |11111111|11111111|11101100               ffffec  [24]
    (145)  |11111111|11111111|11101101               ffffed  [24]
    (148)  |11111111|11111111|11101110               ffffee  [24]
    (159)  |11111111|11111111|11101111               ffffef  [24]
    (171)  |11111111|11111111|11110000               fffff0  [24]
    (206)  |11111111|11111111|11110001               fffff1  [24]
    (215)  |11111111|11111111|11110010               fffff2  [24]
    (225)  |11111111|11111111|11110011               fffff3  [24]
    (236)  |11111111|11111111|11110100               fffff4  [24]
    (237)  |11111111|11111111|11110101               fffff5  [24]
    (199)  |11111111|11111111|11110110|0            1ffffec  [25]
    (207)  |11111111|11111111|11110110|1            1ffffed  [25]
    (234)  |11111111|11111111|11110111|0            1ffffee  [25]
    (235)  |11111111|11111111|11110111|1            1ffffef  [25]
    (192)  |11111111|11111111|11111000|00           3ffffe0  [26]
    (193)  |11111111|11111111|11111000|01           3ffffe1  [26]
    (200)  |11111111|11111111|11111000|10           3ffffe2  [26]
    (201)  |11111111|11111111|11111000|11           3ffffe3  [26]
    (202)  |11111111|11111111|11111001|00           3ffffe4  [26]
    (205)  |11111111|11111111|11111001|01           3ffffe5  [26]
    (210)  |11111111|11111111|11111001|10           3ffffe6  [26]
    (213)  |11111111|11111111|11111001|11           3ffffe7  [26]
    (218)  |11111111|11111111|11111010|00           3ffffe8  [26]
    (219)  |11111111|11111111|11111010|01           3ffffe9  [26]
    (238)  |11111111|11111111|11111010|10           3ffffea  [26]
    (240)  |11111111|11111111|11111010|11           3ffffeb  [26]
    (242)  |11111111|11111111|11111011|00           3ffffec  [26]
    (243)  |11111111|11111111|11111011|01           3ffffed  [26]
    (255)  |11111111|11111111|11111011|10           3ffffee  [26]
    (203)  |11111111|11111111|11111011|110          7ffffde  [27]
    (204)  |11111111|11111111|11111011|111          7ffffdf  [27]
    (211)  |11111111|11111111|11111100|000          7ffffe0  [27]
    (212)  |11111111|11111111|11111100|001          7ffffe1  [27]
    (214)  |11111111|11111111|11111100|010          7ffffe2  [27]
    (221)  |11111111|11111111|11111100|011          7ffffe3  [27]
    (222)  |11111111|11111111|11111100|100          7ffffe4  [27]
    (223)  |11111111|11111111|11111100|101          7ffffe5  [27]
    (241)  |11111111|11111111|11111100|110          7ffffe6  [27]
    (244)  |11111111|11111111|11111100|111          7ffffe7  [27]
    (245)  |11111111|11111111|11111101|000          7ffffe8  [27]
    (246)  |11111111|11111111|11111101|001          7ffffe9  [27]
    (247)  |11111111|11111111|11111101|010          7ffffea  [27]
    (248)  |11111111|11111111|11111101|011          7ffffeb  [27]
    (250)  |11111111|11111111|11111101|100          7ffffec  [27]
    (251)  |11111111|11111111|11111101|101          7ffffed  [27]
    (252)  |11111111|11111111|11111101|110          7ffffee  [27]
    (253)  |11111111|11111111|11111101|111          7ffffef  [27]
    (254)  |11111111|11111111|11111110|000          7fffff0  [27]
    (  2)  |11111111|11111111|11111110|0010         fffffe2  [28]
    (  3)  |11111111|11111111|11111110|0011         fffffe3  [28]
    (  4)  |11111111|11111111|11111110|0100         fffffe4  [28]
    (  5)  |11111111|11111111|11111110|0101         fffffe5  [28]
    (  6)  |11111111|11111111|11111110|0110         fffffe6  [28]
    (  7)  |11111111|11111111|11111110|0111         fffffe7  [28]
    (  8)  |11111111|11111111|11111110|1000         fffffe8  [28]
    ( 11)  |11111111|11111111|11111110|1001         fffffe9  [28]
    ( 12)  |11111111|11111111|11111110|1010         fffffea  [28]
    ( 14)  |11111111|11111111|11111110|1011         fffffeb  [28]
    ( 15)  |11111111|11111111|11111110|1100         fffffec  [28]
    ( 16)  |11111111|11111111|11111110|1101         fffffed  [28]
    ( 17)  |11111111|11111111|11111110|1110         fffffee  [28]
    ( 18)  |11111111|11111111|11111110|1111         fffffef  [28]
    ( 19)  |11111111|11111111|11111111|0000         ffffff0  [28]
    ( 20)  |11111111|11111111|11111111|0001         ffffff1  [28]
    ( 21)  |11111111|11111111|11111111|0010         ffffff2  [28]
    ( 23)  |11111111|11111111|11111111|0011         ffffff3  [28]
    ( 24)  |11111111|11111111|11111111|0100         ffffff4  [28]
    ( 25)  |11111111|11111111|11111111|0101         ffffff5  [28]
    ( 26)  |11111111|11111111|11111111|0110         ffffff6  [28]
    ( 27)  |11111111|11111111|11111111|0111         ffffff7  [28]
    ( 28)  |11111111|11111111|11111111|1000         ffffff8  [28]
    ( 29)  |11111111|11111111|11111111|1001         ffffff9  [28]
    ( 30)  |11111111|11111111|11111111|1010         ffffffa  [28]
    ( 31)  |11111111|11111111|11111111|1011         ffffffb  [28]
    (127)  |11111111|11111111|11111111|1100         ffffffc  [28]
    (220)  |11111111|11111111|11111111|1101         ffffffd  [28]
    (249)  |11111111|11111111|11111111|1110         ffffffe  [28]
    ( 10)  |11111111|11111111|11111111|111100      3ffffffc  [30]
    ( 13)  |11111111|11111111|11111111|111101      3ffffffd  [30]
    ( 22)  |11111111|11111111|11111111|111110      3ffffffe  [30]
    (256)  |11111111|11111111|11111111|111111      3fffffff  [30]
*/

struct {
	uint8_t decoded;
	uint8_t bitcount;
	uint32_t encoded;
} hpack_huff_dec[] = {
    {48, 5, 0x0},
    {49, 5, 0x1},
    {50, 5, 0x2},
    {97, 5, 0x3},
    {99, 5, 0x4},
    {101, 5, 0x5},
    {105, 5, 0x6},
    {111, 5, 0x7},
    {115, 5, 0x8},
    {116, 5, 0x9},
    {32, 6, 0x14},
    {37, 6, 0x15},
    {45, 6, 0x16},
    {46, 6, 0x17},
    {47, 6, 0x18},
    {51, 6, 0x19},
    {52, 6, 0x1a},
    {53, 6, 0x1b},
    {54, 6, 0x1c},
    {55, 6, 0x1d},
    {56, 6, 0x1e},
    {57, 6, 0x1f},
    {61, 6, 0x20},
    {65, 6, 0x21},
    {95, 6, 0x22},
    {98, 6, 0x23},
    {100, 6, 0x24},
    {102, 6, 0x25},
    {103, 6, 0x26},
    {104, 6, 0x27},
    {108, 6, 0x28},
    {109, 6, 0x29},
    {110, 6, 0x2a},
    {112, 6, 0x2b},
    {114, 6, 0x2c},
    {117, 6, 0x2d},
    {58, 7, 0x5c},
    {66, 7, 0x5d},
    {67, 7, 0x5e},
    {68, 7, 0x5f},
    {69, 7, 0x60},
    {70, 7, 0x61},
    {71, 7, 0x62},
    {72, 7, 0x63},
    {73, 7, 0x64},
    {74, 7, 0x65},
    {75, 7, 0x66},
    {76, 7, 0x67},
    {77, 7, 0x68},
    {78, 7, 0x69},
    {79, 7, 0x6a},
    {80, 7, 0x6b},
    {81, 7, 0x6c},
    {82, 7, 0x6d},
    {83, 7, 0x6e},
    {84, 7, 0x6f},
    {85, 7, 0x70},
    {86, 7, 0x71},
    {87, 7, 0x72},
    {89, 7, 0x73},
    {106, 7, 0x74},
    {107, 7, 0x75},
    {113, 7, 0x76},
    {118, 7, 0x77},
    {119, 7, 0x78},
    {120, 7, 0x79},
    {121, 7, 0x7a},
    {122, 7, 0x7b},
    {38, 8, 0xf8},
    {42, 8, 0xf9},
    {44, 8, 0xfa},
    {59, 8, 0xfb},
    {88, 8, 0xfc},
    {90, 8, 0xfd},
    {33, 10, 0x3f8},
    {34, 10, 0x3f9},
    {40, 10, 0x3fa},
    {41, 10, 0x3fb},
    {63, 10, 0x3fc},
    {39, 11, 0x7fa},
    {43, 11, 0x7fb},
    {124, 11, 0x7fc},
    {35, 12, 0xffa},
    {62, 12, 0xffb},
    {0, 13, 0x1ff8},
    {36, 13, 0x1ff9},
    {64, 13, 0x1ffa},
    {91, 13, 0x1ffb},
    {93, 13, 0x1ffc},
    {126, 13, 0x1ffd},
    {94, 14, 0x3ffc},
    {125, 14, 0x3ffd},
    {60, 15, 0x7ffc},
    {96, 15, 0x7ffd},
    {123, 15, 0x7ffe},
    {92, 19, 0x7fff0},
    {195, 19, 0x7fff1},
    {208, 19, 0x7fff2},
    {128, 20, 0xfffe6},
    {130, 20, 0xfffe7},
    {131, 20, 0xfffe8},
    {162, 20, 0xfffe9},
    {184, 20, 0xfffea},
    {194, 20, 0xfffeb},
    {224, 20, 0xfffec},
    {226, 20, 0xfffed},
    {153, 21, 0x1fffdc},
    {161, 21, 0x1fffdd},
    {167, 21, 0x1fffde},
    {172, 21, 0x1fffdf},
    {176, 21, 0x1fffe0},
    {177, 21, 0x1fffe1},
    {179, 21, 0x1fffe2},
    {209, 21, 0x1fffe3},
    {216, 21, 0x1fffe4},
    {217, 21, 0x1fffe5},
    {227, 21, 0x1fffe6},
    {229, 21, 0x1fffe7},
    {230, 21, 0x1fffe8},
    {129, 22, 0x3fffd2},
    {132, 22, 0x3fffd3},
    {133, 22, 0x3fffd4},
    {134, 22, 0x3fffd5},
    {136, 22, 0x3fffd6},
    {146, 22, 0x3fffd7},
    {154, 22, 0x3fffd8},
    {156, 22, 0x3fffd9},
    {160, 22, 0x3fffda},
    {163, 22, 0x3fffdb},
    {164, 22, 0x3fffdc},
    {169, 22, 0x3fffdd},
    {170, 22, 0x3fffde},
    {173, 22, 0x3fffdf},
    {178, 22, 0x3fffe0},
    {181, 22, 0x3fffe1},
    {185, 22, 0x3fffe2},
    {186, 22, 0x3fffe3},
    {187, 22, 0x3fffe4},
    {189, 22, 0x3fffe5},
    {190, 22, 0x3fffe6},
    {196, 22, 0x3fffe7},
    {198, 22, 0x3fffe8},
    {228, 22, 0x3fffe9},
    {232, 22, 0x3fffea},
    {233, 22, 0x3fffeb},
    {1, 23, 0x7fffd8},
    {135, 23, 0x7fffd9},
    {137, 23, 0x7fffda},
    {138, 23, 0x7fffdb},
    {139, 23, 0x7fffdc},
    {140, 23, 0x7fffdd},
    {141, 23, 0x7fffde},
    {143, 23, 0x7fffdf},
    {147, 23, 0x7fffe0},
    {149, 23, 0x7fffe1},
    {150, 23, 0x7fffe2},
    {151, 23, 0x7fffe3},
    {152, 23, 0x7fffe4},
    {155, 23, 0x7fffe5},
    {157, 23, 0x7fffe6},
    {158, 23, 0x7fffe7},
    {165, 23, 0x7fffe8},
    {166, 23, 0x7fffe9},
    {168, 23, 0x7fffea},
    {174, 23, 0x7fffeb},
    {175, 23, 0x7fffec},
    {180, 23, 0x7fffed},
    {182, 23, 0x7fffee},
    {183, 23, 0x7fffef},
    {188, 23, 0x7ffff0},
    {191, 23, 0x7ffff1},
    {197, 23, 0x7ffff2},
    {231, 23, 0x7ffff3},
    {239, 23, 0x7ffff4},
    {9, 24, 0xffffea},
    {142, 24, 0xffffeb},
    {144, 24, 0xffffec},
    {145, 24, 0xffffed},
    {148, 24, 0xffffee},
    {159, 24, 0xffffef},
    {171, 24, 0xfffff0},
    {206, 24, 0xfffff1},
    {215, 24, 0xfffff2},
    {225, 24, 0xfffff3},
    {236, 24, 0xfffff4},
    {237, 24, 0xfffff5},
    {199, 25, 0x1ffffec},
    {207, 25, 0x1ffffed},
    {234, 25, 0x1ffffee},
    {235, 25, 0x1ffffef},
    {192, 26, 0x3ffffe0},
    {193, 26, 0x3ffffe1},
    {200, 26, 0x3ffffe2},
    {201, 26, 0x3ffffe3},
    {202, 26, 0x3ffffe4},
    {205, 26, 0x3ffffe5},
    {210, 26, 0x3ffffe6},
    {213, 26, 0x3ffffe7},
    {218, 26, 0x3ffffe8},
    {219, 26, 0x3ffffe9},
    {238, 26, 0x3ffffea},
    {240, 26, 0x3ffffeb},
    {242, 26, 0x3ffffec},
    {243, 26, 0x3ffffed},
    {255, 26, 0x3ffffee},
    {203, 27, 0x7ffffde},
    {204, 27, 0x7ffffdf},
    {211, 27, 0x7ffffe0},
    {212, 27, 0x7ffffe1},
    {214, 27, 0x7ffffe2},
    {221, 27, 0x7ffffe3},
    {222, 27, 0x7ffffe4},
    {223, 27, 0x7ffffe5},
    {241, 27, 0x7ffffe6},
    {244, 27, 0x7ffffe7},
    {245, 27, 0x7ffffe8},
    {246, 27, 0x7ffffe9},
    {247, 27, 0x7ffffea},
    {248, 27, 0x7ffffeb},
    {250, 27, 0x7ffffec},
    {251, 27, 0x7ffffed},
    {252, 27, 0x7ffffee},
    {253, 27, 0x7ffffef},
    {254, 27, 0x7fffff0},
    {2, 28, 0xfffffe2},
    {3, 28, 0xfffffe3},
    {4, 28, 0xfffffe4},
    {5, 28, 0xfffffe5},
    {6, 28, 0xfffffe6},
    {7, 28, 0xfffffe7},
    {8, 28, 0xfffffe8},
    {11, 28, 0xfffffe9},
    {12, 28, 0xfffffea},
    {14, 28, 0xfffffeb},
    {15, 28, 0xfffffec},
    {16, 28, 0xfffffed},
    {17, 28, 0xfffffee},
    {18, 28, 0xfffffef},
    {19, 28, 0xffffff0},
    {20, 28, 0xffffff1},
    {21, 28, 0xffffff2},
    {23, 28, 0xffffff3},
    {24, 28, 0xffffff4},
    {25, 28, 0xffffff5},
    {26, 28, 0xffffff6},
    {27, 28, 0xffffff7},
    {28, 28, 0xffffff8},
    {29, 28, 0xffffff9},
    {30, 28, 0xffffffa},
    {31, 28, 0xffffffb},
    {127, 28, 0xffffffc},
    {220, 28, 0xffffffd},
    {249, 28, 0xffffffe},
    {10, 30, 0x3ffffffc},
    {13, 30, 0x3ffffffd},
    {22, 30, 0x3ffffffe},
    {(uint8_t)256, 30, 0x3fffffff} /* filling/termination */
};

/* highest value with 5, 6, 7, ... 28, 29, 30 and all (32) bits */
uint32_t hpack_huff_end_code[] = {0x9,       0x2d,       0x7b,       0xfd,
                                  0,         0x3fc,      0x7fc,      0xffb,
                                  0x1ffd,    0x3ffd,     0x7ffe,     0,
                                  0,         0,          0x7fff2,    0xfffed,
                                  0x1fffe8,  0x3fffeb,   0x7ffff4,   0xfffff5,
                                  0x1ffffef, 0x3ffffee,  0x7fffff0,  0xffffffe,
                                  0,         0x3ffffffe, 0xFFFFFFFFu};

/* lowest index with 5, 6, 7, ... 28, 29, 30 and all (32) bits */
uint8_t hpack_huff_start_index[] = {0,   10,  36,  68,  0,   74,  79, 82,  84,
                                    90,  92,  0,   0,   0,   95,  98, 106, 119,
                                    145, 174, 186, 190, 205, 224, 0,  253, 0};


/* Function to decode an integer from a HPACK encoded block */
/* Integers have a variable size encoding, according to the RFC.
 * The integer starts at index *i, idx_mask masks the available bits in
 * the first byte. The index *i is advanced until the end of the
 * encoded integer.
 */
static uint64_t
hpack_getnum(const uint8_t *buf,
             int *i,
             uint8_t idx_mask,
             struct mg_context *ctx)
{
	uint64_t num = (buf[*i] & idx_mask);

	(void)ctx;

	if (num == idx_mask) {
		/* Algorithm from https://tools.ietf.org/html/rfc7541#section-5.1 */
		uint32_t M = 0;
		do {
			(*i)++;
			num = num + ((buf[*i] & 0x7F) << M);
			M += 7;
		} while ((buf[*i] & 0x80) == 0x80);
	}

	(*i)++;
	return num;
}


/* Function to decode a string from a HPACK encoded block */
/* Strings have a variable size and can be either encoded directly (8 bits
 * per char), or using huffman encoding (variable bits per char).
 * The string starts at index *i. This index is advanced until the end of
 * the encoded string.
 */
static char *
hpack_decode(const uint8_t *buf, int *i, struct mg_context *ctx)
{
	uint64_t byte_len64;
	int byte_len;
	int bit_len;
	uint8_t is_huff = ((buf[*i] & 0x80) == 0x80);

	/* Get length of string in bytes */
	byte_len64 = hpack_getnum(buf, i, 0x7f, ctx);
	if (byte_len64 > 1024) {
		/* TODO */
		return NULL;
	}
	byte_len = (int)byte_len64;
	bit_len = byte_len * 8;

	/* Now read the string */
	if (!is_huff) {
		/* Not huffman encoded: Copy directly */
		char *result = mg_malloc_ctx(byte_len + 1, ctx);
		if (result) {
			memcpy(result, buf + (*i), byte_len);
			result[byte_len] = 0;
		}
		(*i) += byte_len;
		return result;

	} else {
		/* Huffman encoded: need to decode bitwise */
		const uint8_t *pData =
		    buf + (*i);           /* begin pointer of bit input string */
		int bitRead = 0;          /* number of encoded bits read */
		uint32_t bytesStored = 0; /* number of decoded bytes stored */
		uint8_t str[2048];        /* storage buffer for decoded string */

		for (;;) {
			uint32_t accu = 0; /* accu register: collect bits */
			uint8_t bc = 0;    /* number of bits collected */
			int n;

			/* Collect bits in this loop, until we have a valid huff code in
			 * accu */
			do {
				accu <<= 1;
				accu |= (pData[bitRead / 8] >> (7 - (bitRead & 7))) & 1;
				bitRead++;
				bc++;
				if (bitRead > bit_len) {
					/* We used all bits. Return the decoded string. */
					str[bytesStored] = 0; /* Terminate string */
					(*i) += byte_len;     /* Advance parsing index */
					return mg_strdup_ctx((char *)str,
					                     ctx); /* Return a string copy */
				}
			} while ((bc < 5) || (accu > hpack_huff_end_code[bc - 5]));

			/* Find matching code in huffman encoding table */
			for (n = hpack_huff_start_index[bc - 5]; n < 256; n++) {
				if (accu == hpack_huff_dec[n].encoded) {
					str[bytesStored] = hpack_huff_dec[n].decoded;
					bytesStored++;
					break;
				}
			}
		}
	}
}


static void
append_bits(uint8_t *target,
            uint32_t offset,
            uint32_t value,
            uint8_t value_bits)
{
	uint32_t offset_bytes = offset / 8;
	uint32_t offset_bits = offset % 8;
	uint32_t remaining_bits, ac;

	value &= ~(0xFFFFFFFF << value_bits);

	remaining_bits = 8 - offset_bits;

	if (value_bits <= remaining_bits) {
		ac = value << (remaining_bits - value_bits);
		target[offset_bytes] |= ac;
		return;
	}

	ac = value >> (value_bits - remaining_bits);
	target[offset_bytes] |= ac;
	append_bits(target,
	            offset + remaining_bits,
	            value,
	            value_bits - remaining_bits);
}


static int
hpack_encode(uint8_t *store, const char *load, int lower)
{
	uint32_t nohuff_len = strlen(load);

	uint32_t len_bits = 0;
	uint32_t len_bytes;
	uint32_t spare_bits;
	uint32_t i;

	memset(store, 0, nohuff_len + 1);

	for (i = 0; i < nohuff_len; i++) {
		uint8_t b = (uint8_t)((char)(lower ? tolower(load[i]) : load[i]));
		int idx;

		for (idx = 0; idx <= 255; idx++) {
			if (hpack_huff_dec[idx].decoded == b) {
				append_bits((uint8_t *)store + 1,
				            len_bits,
				            hpack_huff_dec[idx].encoded,
				            hpack_huff_dec[idx].bitcount);
				len_bits += hpack_huff_dec[idx].bitcount;
				break;
			}
		}
	}

	len_bytes = (len_bits + 7) / 8;
	spare_bits = len_bytes * 8 - len_bits;
	if (spare_bits) {
		append_bits((uint8_t *)store + 1, len_bits, 0xFFFFFFFF, spare_bits);
	}

	if (len_bytes >= 127) {
		// TODO: Shift string and encode len in more bytes
		return 0;
	}
	*store = 0x80 + (uint8_t)len_bytes;

	if ((len_bytes >= nohuff_len) && (0)) {
		*store = (uint8_t)nohuff_len;
		if (lower) {
			for (i = 1; i <= nohuff_len; i++) {
				store[i] = tolower(load[i]);
			}
		} else {
			memcpy(store + 1, load, nohuff_len);
		}
		return nohuff_len + 1;
	} else {
		/*
		int i = 0;
		char *test = hpack_decode(store, &i, NULL);
		i = i; // breakpoint for debugging / testing
		*/
	}

	return len_bytes + 1;
}


/***********************************************************************/
/***  HTTP 2                                                         ***/
/***********************************************************************/


static const char http2_pri[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
static unsigned char http2_pri_len = 24; /* = strlen(http2_pri) */


/* Read and check the HTTP/2 primer/preface:
 * See https://tools.ietf.org/html/rfc7540#section-3.5 */
static int
is_valid_http2_primer(struct mg_connection *conn)
{
	size_t pri_len = http2_pri_len;
	char buf[32];

	if (pri_len > sizeof(buf)) {
		/* Should never be reached - the RFC primer has 24 bytes */
		return 0;
	}
	int read_pri_len = mg_read(conn, buf, pri_len);
	if ((read_pri_len != (int)pri_len)
	    || (0 != memcmp(buf, http2_pri, pri_len))) {
		return 0;
	}
	return 1;
}


#define mg_xwrite(conn, data, len)                                             \
	push_all((conn)->phys_ctx,                                                 \
	         NULL,                                                             \
	         (conn)->client.sock,                                              \
	         (conn)->ssl,                                                      \
	         (const char *)(data),                                             \
	         (int)(len));


static void
http2_settings_acknowledge(struct mg_connection *conn)
{
	unsigned char http2_set_ackn_frame[9] = {0, 0, 0, 4, 1, 0, 0, 0, 0};

	DEBUG_TRACE("%s", "Sending settings frame");
	mg_xwrite(conn, http2_set_ackn_frame, 9);
}


struct http2_settings {
	uint32_t settings_header_table_size;
	uint32_t settings_enable_push;
	uint32_t settings_max_concurrent_streams;
	uint32_t settings_initial_window_size;
	uint32_t settings_max_frame_size;
	uint32_t settings_max_header_list_size;
};


const struct http2_settings http2_default_settings =
    {4096, 1, UINT32_MAX, 65535, 16384, UINT32_MAX};

const struct http2_settings http2_civetweb_server_settings =
    {4096, 0, 100, 65535, 16384, 65535};


enum {
	HTTP2_ERR_NO_ERROR = 0,
	HTTP2_ERR_PROTOCOL_ERROR,
	HTTP2_ERR_INTERNAL_ERROR,
	HTTP2_ERR_FLOW_CONTROL_ERROR,
	HTTP2_ERR_SETTINGS_TIMEOUT,
	HTTP2_ERR_STREAM_CLOSED,
	HTTP2_ERR_FRAME_SIZE_ERROR,
	HTTP2_ERR_REFUSED_STREAM,
	HTTP2_ERR_CANCEL,
	HTTP2_ERR_COMPRESSION_ERROR,
	HTTP2_ERR_CONNECT_ERROR,
	HTTP2_ERR_ENHANCE_YOUR_CALM,
	HTTP2_ERR_INADEQUATE_SECURITY,
	HTTP2_ERR_HTTP_1_1_REQUIRED
};


static void
http2_send_settings(struct mg_connection *conn,
                    const struct http2_settings *set)
{
	uint16_t id;
	uint32_t data;
	uint8_t http2_settings_frame[9] = {0, 0, 36, 4, 0, 0, 0, 0, 0};
	mg_xwrite(conn, http2_settings_frame, 9);

	id = htons(1);
	data = htonl(set->settings_header_table_size);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	id = htons(1);
	data = htonl(set->settings_enable_push);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	id = htons(1);
	data = htonl(set->settings_max_concurrent_streams);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	id = htons(1);
	data = htonl(set->settings_initial_window_size);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	id = htons(1);
	data = htonl(set->settings_max_frame_size);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	id = htons(1);
	data = htonl(set->settings_max_header_list_size);
	mg_xwrite(conn, &id, 2);
	mg_xwrite(conn, &data, 4);

	DEBUG_TRACE("%s", "HTTP2 settings sent");
}


static int
http2_send_response_headers(struct mg_connection *conn)
{
	unsigned char http2_header_frame[9] = {0, 0, 0, 1, 4, 0, 0, 0, 0};
	uint8_t header_bin[1024];
	uint16_t header_len = 0;
	int has_date = 0;
	int has_connection = 0;
	int i;

	if ((conn->status_code < 100) || (conn->status_code > 999)) {
		/* Invalid status: Set status to "Internal Server Error" */
		conn->status_code = 500;
	}

	switch (conn->status_code) {
	case 200:
		header_bin[header_len++] = 0x88;
		break;
	case 204:
		header_bin[header_len++] = 0x89;
		break;
	case 206:
		header_bin[header_len++] = 0x8A;
		break;
	case 304:
		header_bin[header_len++] = 0x8B;
		break;
	case 400:
		header_bin[header_len++] = 0x8C;
		break;
	case 404:
		header_bin[header_len++] = 0x8D;
		break;
	case 500:
		header_bin[header_len++] = 0x8E;
		break;
	default:
		header_bin[header_len++] = 0x48;
		header_bin[header_len++] = 0x03;
		header_bin[header_len++] = 0x30 + (conn->status_code / 100);
		header_bin[header_len++] = 0x30 + ((conn->status_code / 10) % 10);
		header_bin[header_len++] = 0x30 + (conn->status_code % 10);
		break;
	}

	/* Add all headers */
	for (i = 0; i < conn->response_info.num_headers; i++) {
		uint16_t predef = 0;
		uint16_t j;

		/* Filter headers not valid in HTTP/2 */
		if (!mg_strcasecmp("Connection",
		                   conn->response_info.http_headers[i].name)) {
			has_connection = 1;
			continue; /* do not send */
		}

		/* Check if this header is known in HPACK (static table index 15 to 61)
		 * see https://tools.ietf.org/html/rfc7541#appendix-A */
		for (j = 15; j <= 61; j++) {
			if (!mg_strcasecmp(hpack_predefined[j].name,
			                   conn->response_info.http_headers[i].name)) {
				predef = j;
				break;
			}
		}

		if (predef) {
			/* Predefined header found */
			header_bin[header_len++] = 0x40 + predef;
		} else {
			/* Rare header, do not index */
			header_bin[header_len++] = 0x10;
			j = hpack_encode(header_bin + header_len,
			                 conn->response_info.http_headers[i].name,
			                 1);
			header_len += j;
		}

		j = hpack_encode(header_bin + header_len,
		                 conn->response_info.http_headers[i].value,
		                 0);
		header_len += j;

		/* Mark required headers as sent */
		if (!mg_strcasecmp("Date", conn->response_info.http_headers[i].name)) {
			has_date = 1;
		}
	}

	/* Add required headers, if they have not been sent yet */
	if (!has_date) {
		/* Create header frame */
		char date[64];
		uint8_t date_len;
		time_t curtime = time(NULL);

		gmt_time_string(date, sizeof(date), &curtime);
		date_len = (uint8_t)strlen(date);

		header_bin[header_len++] =
		    0x61; /* "Date" predefined HPACK index 33 (0x21) + 0x40 */
		header_bin[header_len++] = date_len;
		memcpy(header_bin + header_len, date, date_len);
		header_len += date_len;
	}

	http2_header_frame[1] = (header_len & 0xFF00) >> 8;
	http2_header_frame[2] = (header_len & 0xFF);

	http2_header_frame[5] = (conn->http2.stream_id & 0xFF000000u) >> 24;
	http2_header_frame[6] = (conn->http2.stream_id & 0xFF0000u) >> 16;
	http2_header_frame[7] = (conn->http2.stream_id & 0xFF00u) >> 8;
	http2_header_frame[8] = (conn->http2.stream_id & 0xFFu);

	/* Send header frame */
	mg_xwrite(conn, http2_header_frame, 9);
	mg_xwrite(conn, header_bin, header_len);

	DEBUG_TRACE("HTTP2 response header sent: stream %u", conn->http2.stream_id);

	return 42; /* TODO */
}


static void
http2_data_frame_head(struct mg_connection *conn,
                      uint32_t frame_size,
                      int is_final)
{
	unsigned char http2_data_frame[9];
	uint32_t stream_id = conn->http2.stream_id;

	http2_data_frame[0] = (frame_size & 0xFF0000) >> 16;
	http2_data_frame[1] = (frame_size & 0xFF00) >> 8;
	http2_data_frame[2] = (frame_size & 0xFF);

	http2_data_frame[3] = 0; /* frame type "DATA" */
	http2_data_frame[4] = (is_final ? 1 : 0);

	http2_data_frame[5] = (stream_id & 0xFF000000u) >> 24;
	http2_data_frame[6] = (stream_id & 0xFF0000u) >> 16;
	http2_data_frame[7] = (stream_id & 0xFF00u) >> 8;
	http2_data_frame[8] = (stream_id & 0xFFu);

	DEBUG_TRACE("HTTP2 begin data frame: stream %u, frame_size %u (final: %i)",
	            stream_id,
	            frame_size,
	            is_final);

	mg_xwrite(conn, http2_data_frame, 9);
}


static void
http2_send_window(struct mg_connection *conn,
                  uint32_t stream_id,
                  uint32_t window_size)
{
	unsigned char http2_window_frame[9] = {0, 0, 4, 8, 0, 0, 0, 0, 0};
	uint32_t data = htonl(window_size);

	DEBUG_TRACE("HTTP2 send window_size: stream %u, error %u",
	            stream_id,
	            window_size);

	http2_window_frame[5] = (stream_id & 0xFF000000u) >> 24;
	http2_window_frame[6] = (stream_id & 0xFF0000u) >> 16;
	http2_window_frame[7] = (stream_id & 0xFF00u) >> 8;
	http2_window_frame[8] = (stream_id & 0xFFu);
	mg_xwrite(conn, http2_window_frame, 9);
	mg_xwrite(conn, &data, 4);
}


static void
http2_reset_stream(struct mg_connection *conn,
                   uint32_t stream_id,
                   uint32_t error_id)
{
	unsigned char http2_reset_frame[9] = {0, 0, 4, 3, 0, 0, 0, 0, 0};
	uint32_t val = htonl(error_id);

	DEBUG_TRACE("HTTP2 send reset: stream %u, error %u", stream_id, error_id);

	http2_reset_frame[5] = (stream_id & 0xFF000000u) >> 24;
	http2_reset_frame[6] = (stream_id & 0xFF0000u) >> 16;
	http2_reset_frame[7] = (stream_id & 0xFF00u) >> 8;
	http2_reset_frame[8] = (stream_id & 0xFFu);
	mg_xwrite(conn, http2_reset_frame, 9);
	mg_xwrite(conn, &val, 4);
}


static void
http2_must_use_http1(struct mg_connection *conn)
{
	DEBUG_TRACE("HTTP2 not available for this URL (%s)", conn->path_info);
	http2_reset_stream(conn, conn->http2.stream_id, 0xd);
}


/* The HTTP2 implementation collects request headers as array of dynamically
 * allocated string values. This array must be freed once the request is
 * handled.
 * This is different to the HTTP/1.x implementation: For HTTP/1.x, the header
 * list is implemented as pointers into an existing buffer, so free must not
 * be called for HTTP/1.x.
 * Thus free_buffered_request_header_list is in mod_http2.inl.
 */
#if defined(DEBUG)
static int mem_h_count = 0;
static int mem_d_count = 0;
#define CHECK_LEAK_HDR_ALLOC(ptr)                                              \
	DEBUG_TRACE("H NEW %08x (%i): %s",                                         \
	            (uint32_t)ptr,                                                 \
	            ++mem_h_count,                                                 \
	            (const char *)ptr)
#define CHECK_LEAK_HDR_FREE(ptr)                                               \
	DEBUG_TRACE("H DEL %08x (%i): %s",                                         \
	            (uint32_t)ptr,                                                 \
	            --mem_h_count,                                                 \
	            (const char *)ptr)
#define CHECK_LEAK_DYN_ALLOC(ptr)                                              \
	DEBUG_TRACE("D NEW %08x (%i): %s",                                         \
	            (uint32_t)ptr,                                                 \
	            ++mem_d_count,                                                 \
	            (const char *)ptr)
#define CHECK_LEAK_DYN_FREE(ptr)                                               \
	DEBUG_TRACE("D DEL %08x (%i): %s",                                         \
	            (uint32_t)ptr,                                                 \
	            --mem_d_count,                                                 \
	            (const char *)ptr)
#else
#define CHECK_LEAK_HDR_ALLOC(ptr)
#define CHECK_LEAK_HDR_FREE(ptr)
#define CHECK_LEAK_DYN_ALLOC(ptr)
#define CHECK_LEAK_DYN_FREE(ptr)
#endif


/* The dynamic header table may be resized on a HTTP2 client request.
 * A tablesize=0 will free all memory.
 */
static void
purge_dynamic_header_table(struct mg_connection *conn, uint32_t tableSize)
{
	DEBUG_TRACE("HTTP2 dynamic header table set to %u", tableSize);
	while (conn->http2.dyn_table_size > tableSize) {
		conn->http2.dyn_table_size--;

		CHECK_LEAK_DYN_FREE(
		    conn->http2.dyn_table[conn->http2.dyn_table_size].name);
		CHECK_LEAK_DYN_FREE(
		    conn->http2.dyn_table[conn->http2.dyn_table_size].value);

		mg_free((void *)conn->http2.dyn_table[conn->http2.dyn_table_size].name);
		conn->http2.dyn_table[conn->http2.dyn_table_size].name = 0;
		mg_free(
		    (void *)conn->http2.dyn_table[conn->http2.dyn_table_size].value);
		conn->http2.dyn_table[conn->http2.dyn_table_size].value = 0;
	}
}


/* Internal function to free request header list.
 * Not to be confused with the response header list.
 */
static void
free_buffered_request_header_list(struct mg_connection *conn)
{
	while (conn->request_info.num_headers > 0) {
		conn->request_info.num_headers--;

		CHECK_LEAK_HDR_FREE(
		    conn->request_info.http_headers[conn->request_info.num_headers]
		        .name);
		CHECK_LEAK_HDR_FREE(
		    conn->request_info.http_headers[conn->request_info.num_headers]
		        .value);

		mg_free((void *)conn->request_info
		            .http_headers[conn->request_info.num_headers]
		            .name);
		conn->request_info.http_headers[conn->request_info.num_headers].name =
		    0;
		mg_free((void *)conn->request_info
		            .http_headers[conn->request_info.num_headers]
		            .value);
		conn->request_info.http_headers[conn->request_info.num_headers].value =
		    0;
	}
}


/* HTTP2 requires a different handling loop */
static void
handle_http2(struct mg_connection *conn)
{
	unsigned char http2_frame_head[9];
	uint32_t http2_frame_size;
	uint8_t http2_frame_type;
	uint8_t http2_frame_flags;
	uint32_t http2_frame_stream_id;
	uint32_t http_window_length = 0;
	int bytes_read;
	uint8_t *buf;
	int my_settings_accepted = 0;
	int my_settings_sent;
	const char *my_hpack_headers[128];

	struct http2_settings client_settings = http2_default_settings;
	struct http2_settings server_settings = http2_default_settings;

	/* Send own settings */
	http2_send_settings(conn, &http2_civetweb_server_settings);
	my_settings_sent = 1;
	// http2_send_window(conn, 0, /* 0x3fff0001 */ 1024*1024);

	/* initialize hpack header table with predefined header fields */
	memset((void *)my_hpack_headers, 0, sizeof(my_hpack_headers));
	memcpy((void *)my_hpack_headers,
	       hpack_predefined,
	       sizeof(hpack_predefined));

	buf = (uint8_t *)mg_malloc_ctx(server_settings.settings_max_frame_size,
	                               conn->phys_ctx);
	if (!buf) {
		/* Out of memory */
		DEBUG_TRACE("%s", "Out of memory for HTTP2 frame");
		return;
	}

	for (;;) {
		/* HTTP/2 is handled frame by frame */
		int frame_is_end_stream = 0;
		int frame_is_end_headers = 0;
		int frame_is_padded = 0;
		int frame_is_priority = 0;

		bytes_read = mg_read(conn, http2_frame_head, sizeof(http2_frame_head));
		if (bytes_read != sizeof(http2_frame_head)) {
			/* TODO: errormsg */
			goto clean_http2;
		}

		/* Extract data from frame header */
		http2_frame_size = ((uint32_t)http2_frame_head[0] * 0x10000u)
		                   + ((uint32_t)http2_frame_head[1] * 0x100u)
		                   + ((uint32_t)http2_frame_head[2]);
		http2_frame_type = http2_frame_head[3];
		http2_frame_flags = http2_frame_head[4];
		http2_frame_stream_id = ((uint32_t)http2_frame_head[5] * 0x1000000u)
		                        + ((uint32_t)http2_frame_head[6] * 0x10000u)
		                        + ((uint32_t)http2_frame_head[7] * 0x100u)
		                        + ((uint32_t)http2_frame_head[8]);

		frame_is_end_stream = (0 != (http2_frame_flags & 0x01));
		frame_is_end_headers = (0 != (http2_frame_flags & 0x04));
		frame_is_padded = (0 != (http2_frame_flags & 0x08));
		frame_is_priority = (0 != (http2_frame_flags & 0x20));

		if (http2_frame_size > server_settings.settings_max_frame_size) {
			/* TODO: Error Message */
			DEBUG_TRACE("HTTP2 frame too large (%lu)",
			            (unsigned long)http2_frame_size);
			goto clean_http2;
		}
		bytes_read = mg_read(conn, buf, http2_frame_size);
		if (bytes_read != (int)http2_frame_size) {
			/* TODO: Error Message - or read again? */
			DEBUG_TRACE("HTTP2 read error (%li != %li)",
			            (signed long int)bytes_read,
			            (signed long int)http2_frame_size);
			goto clean_http2;
		}

		DEBUG_TRACE("HTTP2 frame type %u, size %u, stream %u, flags %02x",
		            http2_frame_type,
		            http2_frame_size,
		            http2_frame_stream_id,
		            http2_frame_flags);

		/* Further processing according to frame type. See definition: */
		/* https://tools.ietf.org/html/rfc7540#section-6 */
		switch (http2_frame_type) {

		case 0: /* DATA */
		{
			int i = 0; /* TODO */
			DEBUG_TRACE("%s", "HTTP2 DATA frame?");
		} break;

		case 1: /* HEADERS */
		{
			int i = 0;
			uint8_t padding = 0;
			uint32_t dependency = 0;
			uint8_t weight = 0;
			uint8_t exclusive = 0;

			if (frame_is_padded) {
				padding = buf[i];
				i++;
				DEBUG_TRACE("HTTP2 frame padded by %u bytes", padding);
			}
			if (frame_is_priority) {
				uint32_t val = ((uint32_t)buf[0 + i] * 0x1000000u)
				               + ((uint32_t)buf[1 + i] * 0x10000u)
				               + ((uint32_t)buf[2 + i] * 0x100u)
				               + ((uint32_t)buf[3 + i]);
				dependency = (val & 0x7FFFFFFFu);
				exclusive = ((val & 0x80000000u) != 0);
				weight = buf[4 + i];
				i += 5;
				DEBUG_TRACE(
				    "HTTP2 frame weight %u, dependency %u (exclusive: %i)",
				    weight,
				    dependency,
				    exclusive);
			}

			conn->request_info.num_headers = 0;

			while (i < (int)http2_frame_size - (int)padding) {
				const char *key = 0;
				const char *val = 0;
				uint8_t idx_mask = 0;
				uint8_t value_known = 0;
				uint8_t indexing = 0;
				uint64_t idx = 0;

				/* Classify next entry by checking the bit mask */
				if ((buf[i] & 0x80u) == 0x80u) {
					/* Indexed Header Field Representation:
					 * https://tools.ietf.org/html/rfc7541#section-6.1 */
					idx_mask = 0x7fu;
					value_known = 1;

				} else if ((buf[i] & 0xC0u) == 0x40u) {
					/* Literal Header Field with Incremental Indexing:
					 * https://tools.ietf.org/html/rfc7541#section-6.2.1 */
					idx_mask = 0x3fu;
					indexing = 1;

				} else if ((buf[i] & 0xF0u) == 0x00u) {
					/* Literal Header Field without Indexing:
					 * https://tools.ietf.org/html/rfc7541#section-6.2.2 */
					idx_mask = 0x0fu;

				} else if ((buf[i] & 0xF0u) == 0x10u) {
					/* Literal Header Field Never Indexed:
					 * https://tools.ietf.org/html/rfc7541#section-6.2.3 */
					idx_mask = 0x0fu;

				} else if ((buf[i] & 0xE0u) == 0x20u) {
					uint64_t tableSize;
					/* Dynamic Table Size Update:
					 * https://tools.ietf.org/html/rfc7541#section-6.3 */
					idx_mask = 0x1fu;
					tableSize = hpack_getnum(buf, &i, idx_mask, conn->phys_ctx);

					/* TODO: check if tablesize > allowed table size */

					/* Purge additional table entries */
					purge_dynamic_header_table(conn, (uint32_t)tableSize);

					/* Process next frame */
					continue;

				} else {
					DEBUG_TRACE("HTTP2 unknown start pattern %02x", buf[i]);
					goto clean_http2;
				}

				/* Get the header name table index */
				idx = hpack_getnum(buf, &i, idx_mask, conn->phys_ctx);

				/* Get Header name "key" */
				if (idx == 0) {
					/* Index 0: Header name encoded in following bytes */
					key = hpack_decode(buf, &i, conn->phys_ctx);
					CHECK_LEAK_HDR_ALLOC(key);
				} else if (/*(idx >= 15) &&*/ (idx <= 61)) {
					/* Take key name from predefined header table */
					key = mg_strdup_ctx(hpack_predefined[idx].name,
					                    conn->phys_ctx); /* leak? */
					CHECK_LEAK_HDR_ALLOC(key);
				} else if ((idx >= 62)
				           && ((idx - 61) <= conn->http2.dyn_table_size)) {
					/* Take from dynamic header table */
					uint32_t local_table_idx = (uint32_t)idx - 62;
					key = mg_strdup_ctx(
					    conn->http2.dyn_table[local_table_idx].name,
					    conn->phys_ctx);
					CHECK_LEAK_HDR_ALLOC(key);
				} else {
					/* protocol violation */
					DEBUG_TRACE("HTTP2 invalid index %lu", (unsigned long)idx);
					goto clean_http2;
				}
				/* key is allocated now and must be freed later */

				/* Get header value */
				if (value_known) {
					/* Server must already know the value */
					if (idx <= 61) {
						if (hpack_predefined[idx].value) {
							val = mg_strdup_ctx(hpack_predefined[idx].value,
							                    conn->phys_ctx); /* leak? */
							CHECK_LEAK_HDR_ALLOC(val);
						} else {
							/* protocol violation */
							DEBUG_TRACE("HTTP2 indexed header %lu has no value "
							            "(key: %s)",
							            (unsigned long)idx,
							            key);
							CHECK_LEAK_HDR_FREE(key);
							mg_free((void *)key);
							goto clean_http2;
						}
					} else if ((idx >= 62)
					           && ((idx - 61) <= conn->http2.dyn_table_size)) {
						uint32_t local_table_idx = (uint32_t)idx - 62;
						val = mg_strdup_ctx(
						    conn->http2.dyn_table[local_table_idx].value,
						    conn->phys_ctx);
						CHECK_LEAK_HDR_ALLOC(val);
					} else {
						/* protocol violation */
						DEBUG_TRACE(
						    "HTTP2 indexed header %lu out of range (key: %s)",
						    (unsigned long)idx,
						    key);
						CHECK_LEAK_HDR_FREE(key);
						mg_free((void *)key);
						goto clean_http2;
					}

				} else {
					/* Read value from HTTP2 stream */
					val = hpack_decode(buf, &i, conn->phys_ctx); /* leak? */
					CHECK_LEAK_HDR_ALLOC(val);

					if (indexing) {
						/* Add to index */
						if (conn->http2.dyn_table_size
						    >= HTTP2_DYN_TABLE_SIZE) {
							/* Too many elements */
							DEBUG_TRACE("HTTP2 index table is full (key: %s, "
							            "value: %s)",
							            key,
							            val);

							CHECK_LEAK_HDR_FREE(key);
							CHECK_LEAK_HDR_FREE(val);

							mg_free((void *)key);
							mg_free((void *)val);
							goto clean_http2;
						}

						/* Add to table of dynamic headers */
						conn->http2.dyn_table[conn->http2.dyn_table_size].name =
						    mg_strdup_ctx(key, conn->phys_ctx); /* leak */
						conn->http2.dyn_table[conn->http2.dyn_table_size]
						    .value =
						    mg_strdup_ctx(val, conn->phys_ctx); /* leak */

						CHECK_LEAK_DYN_ALLOC(
						    conn->http2.dyn_table[conn->http2.dyn_table_size]
						        .name);
						CHECK_LEAK_DYN_ALLOC(
						    conn->http2.dyn_table[conn->http2.dyn_table_size]
						        .value);

						conn->http2.dyn_table_size++;

						DEBUG_TRACE("HTTP2 new dynamic header table entry %i "
						            "(key: %s, value: %s)",
						            (int)conn->http2.dyn_table_size,
						            key,
						            val);
					}
				}
				/* val and key are allocated now and must be freed later */
				/* Store these pointers in conn->request_info[].http_headers,
				 * free_buffered_header_list(conn) will clean up later. */

				/* Add header for this request */
				if ((key != NULL) && (val != NULL)
				    && (conn->request_info.num_headers < MG_MAX_HEADERS)) {
					conn->request_info
					    .http_headers[conn->request_info.num_headers]
					    .name = key;
					conn->request_info
					    .http_headers[conn->request_info.num_headers]
					    .value = val;
					conn->request_info.num_headers++;

					/* Some headers need to be stored in the request structure
					 */
					if (!strcmp(":method", key)) {
						conn->request_info.request_method = val;
					} else if (!strcmp(":path", key)) {
						conn->request_info.local_uri = val;
						conn->request_info.request_uri = val;
					} else if (!strcmp(":status", key)) {
						conn->status_code = atoi(val);
					}

					DEBUG_TRACE("HTTP2 request header (key: %s, value: %s)",
					            key,
					            val);

				} else {
					/* - either key or value are NULL (out of memory)
					 * - or the max. number of headers is reached
					 * in both cases free all memory
					 */
					DEBUG_TRACE("%s", "HTTP2 cannot add header");
					CHECK_LEAK_HDR_FREE(key);
					CHECK_LEAK_HDR_FREE(val);

					mg_free((void *)key);
					key = NULL;
					mg_free((void *)val);
					val = NULL;
				}
			}

			/* stream id */
			conn->http2.stream_id = http2_frame_stream_id;

			/* header parsed */
			DEBUG_TRACE("HTTP2 handle_request (stream %u)",
			            http2_frame_stream_id);
			handle_request(conn);

			/* Send "final" frame */
			DEBUG_TRACE("HTTP2 handle_request done (stream %u)",
			            http2_frame_stream_id);
			http2_data_frame_head(conn, 0, 1);
			free_buffered_response_header_list(conn);
			free_buffered_request_header_list(conn);
		} break;

		case 2: /* PRIORITY */
		{
			uint32_t dependStream =
			    ((uint32_t)buf[0] * 0x1000000u) + ((uint32_t)buf[1] * 0x10000u)
			    + ((uint32_t)buf[2] * 0x100u) + ((uint32_t)buf[3]);
			uint8_t weight = buf[4];
			DEBUG_TRACE("HTTP2 priority %u dependent stream %u",
			            weight,
			            dependStream);
		} break;

		case 3: /* RST_STREAM */
		{
			uint32_t errorId =
			    ((uint32_t)buf[0] * 0x1000000u) + ((uint32_t)buf[1] * 0x10000u)
			    + ((uint32_t)buf[2] * 0x100u) + ((uint32_t)buf[3]);
			DEBUG_TRACE("HTTP2 reset with error %u", errorId);
		} break;

		case 4: /* SETTINGS */
			if (http2_frame_stream_id != 0) {
				/* Send protocol error */
				http2_reset_stream(conn,
				                   http2_frame_stream_id,
				                   HTTP2_ERR_PROTOCOL_ERROR);
				DEBUG_TRACE("%s", "HTTP2 received invalid settings frame");
			} else if (http2_frame_flags) {
				/* ACK frame. Do not reply. */
				my_settings_accepted++;
				DEBUG_TRACE("%s", "CivetWeb settings confirmed by peer");
			} else {
				int i;
				for (i = 0; i < (int)http2_frame_size; i += 6) {
					uint16_t id =
					    ((uint16_t)buf[i] * 0x100u) + ((uint16_t)buf[i + 1]);
					uint32_t val = ((uint32_t)buf[i + 2] * 0x1000000u)
					               + ((uint32_t)buf[i + 3] * 0x10000u)
					               + ((uint32_t)buf[i + 4] * 0x100u)
					               + ((uint32_t)buf[i + 5]);
					switch (id) {
					case 1:
						client_settings.settings_header_table_size = val;
						DEBUG_TRACE("Received settings header_table_size: %u",
						            val);
						break;
					case 2:
						client_settings.settings_enable_push = (val != 0);
						DEBUG_TRACE("Received settings enable_push: %u", val);
						break;
					case 3:
						client_settings.settings_max_concurrent_streams = val;
						DEBUG_TRACE(
						    "Received settings max_concurrent_streams: %u",
						    val);
						break;
					case 4:
						client_settings.settings_initial_window_size = val;
						DEBUG_TRACE("Received settings initial_window_size: %u",
						            val);
						break;
					case 5:
						client_settings.settings_max_frame_size = val;
						DEBUG_TRACE("Received settings max_frame_size: %u",
						            val);
						break;
					case 6:
						client_settings.settings_max_header_list_size = val;
						DEBUG_TRACE(
						    "Received settings max_header_list_size: %u", val);
						break;
					default:
						/* Unknown setting. Ignore it. */
						DEBUG_TRACE("Received unknown settings id=%u: %u",
						            id,
						            val);
						break;
					}
				}

				/* Every settings frame must be acknowledged */
				http2_settings_acknowledge(conn);
			}
			break;

		case 5: /* PUSH_PROMISE */
			DEBUG_TRACE("%s", "Push promise not supported");
			break;

		case 6: /* PING */
			if (http2_frame_flags == 0) {
				/* Set "reply" flag, and send same data back */
				DEBUG_TRACE("%s", "Replying to ping");
				http2_frame_head[4] = 1;
				mg_xwrite(conn, http2_frame_head, sizeof(http2_frame_head));
				mg_xwrite(conn, buf, http2_frame_size);
			}
			break;

		case 7: /* GOAWAY */
		{
			uint32_t lastStream =
			    ((uint32_t)buf[0] * 0x1000000u) + ((uint32_t)buf[1] * 0x10000u)
			    + ((uint32_t)buf[2] * 0x100u) + ((uint32_t)buf[3]);
			uint32_t errorId =
			    ((uint32_t)buf[4] * 0x1000000u) + ((uint32_t)buf[5] * 0x10000u)
			    + ((uint32_t)buf[6] * 0x100u) + ((uint32_t)buf[7]);
			; /* followed by debug data */
			uint32_t debugDataLen = http2_frame_size - 8;
			char *debugData = (char *)buf + 8;

			DEBUG_TRACE("HTTP2 goaway stream %u, error %u (%.*s)",
			            lastStream,
			            errorId,
			            debugDataLen,
			            debugData);

		} break;

		case 8: /* WINDOW_UPDATE */
		{
			uint32_t val = ((uint32_t)buf[0] * 0x1000000u)
			               + ((uint32_t)buf[1] * 0x10000u)
			               + ((uint32_t)buf[2] * 0x100u) + ((uint32_t)buf[3]);
			http_window_length = (val & 0x7FFFFFFFu);

			DEBUG_TRACE("HTTP2 window update stream %u, length %u",
			            http2_frame_stream_id,
			            http_window_length);
		} break;

		case 9: /* CONTINUATION */
			DEBUG_TRACE("%s", "HTTP2 Continue");
			break;

		default:
			/* TODO: Error Message */
			DEBUG_TRACE("%s", "Unknown frame type");
			goto clean_http2;
		}
	}

clean_http2:
	DEBUG_TRACE("%s", "HTTP2 free buffer, connection handler finished");
	mg_free(buf);
}


#if 0
static void
HPACK_TEST()
{
	uint64_t test;

	for (test = 0;; test++) {
		char in[32] = {0};
		uint8_t out[32] = {0};
		char *check;
		int i;
		int l;

		memcpy(in, &test, sizeof(test));
		l = hpack_encode(out, in, 0);
		i = 0;
		check = hpack_decode(out, &i, NULL);

		if (strcmp(in, check)) {
			printf("Error\n");
		}
		mg_free(check);
	}
}

static void
HPACK_TABLE_TEST()
{
	int i;

	uint32_t hpack_huff_end_code_expected[32] = { 0 };
	uint8_t hpack_huff_start_index_expected[32] = { 0 };
	int reverse_map[256] = { 0 };

	for (i = 0; i < 256; i++) {
		reverse_map[i] = -1;
	}

	for (i = 0; i < 256; i++) {
		uint8_t bits = hpack_huff_dec[i].bitcount;
		uint8_t dec = hpack_huff_dec[i].decoded;
		if (bits > hpack_huff_dec[i + 1].bitcount) {
			ck_abort_msg("hpack_huff_dec disorder at index %i", i);
		}
		if (hpack_huff_dec[i].encoded & (0xFFFFFFFFul << bits)) {
			ck_abort_msg("hpack_huff_dec bits inconsistent at index %i", i);
		}
		if ((bits < 5) || (bits > 30)) {
			ck_abort_msg("hpack_huff_dec bits out of range at index %i", i);
		}
		if (reverse_map[dec] != -1) {
			ck_abort_msg("hpack_huff_dec duplicate: %i", hpack_huff_dec[i].decoded);
		}
		reverse_map[dec] = i;

		hpack_huff_end_code_expected[bits - 5] = hpack_huff_dec[i].encoded;
	}

	for (i = 255; i >= 0; i--) {
		uint8_t bits = hpack_huff_dec[i].bitcount;
		hpack_huff_start_index_expected[bits - 5] = i;
	}

	for (i = 0; i < 256; i++) {
		if (reverse_map[i] == -1) {
			ck_abort_msg("reverse map at %i mising", i);
		}
	}

	i = sizeof(hpack_huff_start_index) / sizeof(hpack_huff_start_index[0]);
	if (i != 27) {
		ck_abort_msg("hpack_huff_start_index size error: ", i);
	}

	i = sizeof(hpack_huff_end_code) / sizeof(hpack_huff_end_code[0]);
	if (i != 27) {
		ck_abort_msg("hpack_huff_end_code size error: ", i);
	}

	for (i = 0; i < 27; i++) {
		if (hpack_huff_start_index_expected[i] != hpack_huff_start_index[i]) {
			ck_abort_msg("hpack_huff_start_index error at %i", i);
		}
		if (hpack_huff_end_code_expected[i] != hpack_huff_end_code[i]) {
			ck_abort_msg("hpack_huff_end_code error at %i", i);
		}
	}

}
#endif


static void
process_new_http2_connection(struct mg_connection *conn)
{
	if (!is_valid_http2_primer(conn)) {
		/* Primer does not match expectation from RFC.
		 * See https://tools.ietf.org/html/rfc7540#section-3.5 */
		DEBUG_TRACE("%s", "No valid HTTP2 primer");
		mg_send_http_error(conn, 400, "%s", "Invalid HTTP/2 primer");

	} else {
		/* Valid HTTP/2 primer received */
		DEBUG_TRACE("%s", "Start handling HTTP2");
		handle_http2(conn);

		/* Free memory allocated for headers, if not done yet */
		DEBUG_TRACE("%s", "Free remaining HTTP2 header memory");
		free_buffered_response_header_list(conn);
		free_buffered_request_header_list(conn);
		purge_dynamic_header_table(conn, 0);
	}
}
