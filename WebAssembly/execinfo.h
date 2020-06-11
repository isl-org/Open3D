#ifdef __cplusplus
extern "C" {
#endif

int     backtrace(void **, int);
char ** backtrace_symbols(void *const *, int);
void    backtrace_symbols_fd(void *const *, int, int);

#ifdef __cplusplus
}
#endif
