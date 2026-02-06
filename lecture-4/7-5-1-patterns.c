// We want to rewrite this:
if (condition) {

} else {
  do_something();
}

// Into this:
if (!condition) {
  do_something();
} else {
}
