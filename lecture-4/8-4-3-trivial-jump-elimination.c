void f() {
	if (condition) {
		do_something();
		goto label;
	}
label:
}

void f() {
	if (condition)
		do_something();
}
