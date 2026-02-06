# Emit to stdout:
revng clift-opt -o /dev/null --emit-c

# Emit to stdout with PTML tags:
revng clift-opt -o /dev/null --emit-c=ptml

# Emit to ./out.c with PTML tags:
revng clift-opt -o /dev/null --emit-c=output=./out.c
